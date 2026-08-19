import Combine
import Foundation
import HealthKit
import WatchConnectivity
import WatchKit

struct WatchSet: Codable, Identifiable, Equatable {
  let id: String
  var index: Int
  var type: String
  var weightKg: Double?
  var reps: Int?
  var durationSec: Int?
  var rpe: Double? = nil
  var previousWeightKg: Double?
  var previousReps: Int?
  var previousDurationSec: Int?
  var completed: Bool
}

struct WatchExercise: Codable, Identifiable, Equatable {
  let id: String
  let exerciseId: String
  let name: String
  let kind: String
  let note: String?
  var restSec: Int
  var supersetId: String? = nil
  var sets: [WatchSet]
}

struct WatchSetTimerState: Codable, Equatable {
  let sessionExerciseId: String
  let setId: String
  let targetSec: Int
  let startedAt: String
  let endsAt: String
  let pausedRemainingSec: Int?
}

struct WatchRestTimerState: Codable, Equatable {
  let endsAt: Double
  let totalSec: Int
}

struct WatchWorkoutPlan: Codable, Identifiable, Equatable {
  let id: String
  var name: String
  let routineId: String?
  var startedAt: String?
  var endedAt: String?
  var updatedAt: String
  var revision: Int
  var exercises: [WatchExercise]
  var heartRateBpm: Double?
  var activeCalories: Double?
  var activeSetTimer: WatchSetTimerState? = nil
  var activeRestTimer: WatchRestTimerState? = nil
}

struct WatchWorkoutLibrary: Codable {
  let version: Int
  let unit: String
  let updatedAt: String
  let routines: [WatchWorkoutPlan]
  let activeWorkout: WatchWorkoutPlan?
  let activeWorkoutClearedAt: String?
  let activeWorkoutClearedId: String?
  let defaultRestSec: Int
  let catalog: [WatchCatalogExercise]
}

struct WatchCatalogExercise: Codable, Identifiable, Equatable {
  let id: String
  let name: String
  let kind: String
  let primaryMuscle: String
}

struct WatchWorkoutSummary: Equatable {
  let name: String
  let elapsedSec: Int
  let completedSets: Int
  let activeCalories: Int
}

struct WatchCountdown: Equatable {
  let targetSec: Int
  let startedAt: Date
  let endsAt: Date
  let pausedRemaining: Int?
  let exerciseId: String?
  let setId: String?

  init(seconds: Int, exerciseId: String? = nil, setId: String? = nil) {
    targetSec = max(1, seconds)
    startedAt = Date()
    endsAt = startedAt.addingTimeInterval(TimeInterval(max(1, seconds)))
    pausedRemaining = nil
    self.exerciseId = exerciseId
    self.setId = setId
  }

  init(
    targetSec: Int,
    startedAt: Date,
    endsAt: Date,
    pausedRemaining: Int?,
    exerciseId: String?,
    setId: String?
  ) {
    self.targetSec = targetSec
    self.startedAt = startedAt
    self.endsAt = endsAt
    self.pausedRemaining = pausedRemaining
    self.exerciseId = exerciseId
    self.setId = setId
  }

  var paused: Bool { pausedRemaining != nil }
  var remaining: Int { pausedRemaining ?? max(0, Int(ceil(endsAt.timeIntervalSinceNow))) }

  func stopped() -> Self {
    .init(
      targetSec: targetSec,
      startedAt: startedAt,
      endsAt: endsAt,
      pausedRemaining: remaining,
      exerciseId: exerciseId,
      setId: setId
    )
  }

  func resumed() -> Self {
    .init(
      targetSec: targetSec,
      startedAt: startedAt,
      endsAt: Date().addingTimeInterval(TimeInterval(max(1, remaining))),
      pausedRemaining: nil,
      exerciseId: exerciseId,
      setId: setId
    )
  }

  func adjusted(by seconds: Int) -> Self {
    let next = max(0, remaining + seconds)
    if paused {
      return .init(
        targetSec: max(1, targetSec + seconds),
        startedAt: startedAt,
        endsAt: endsAt,
        pausedRemaining: next,
        exerciseId: exerciseId,
        setId: setId
      )
    }
    return .init(
      targetSec: max(1, targetSec + seconds),
      startedAt: startedAt,
      endsAt: Date().addingTimeInterval(TimeInterval(next)),
      pausedRemaining: nil,
      exerciseId: exerciseId,
      setId: setId
    )
  }
}

@MainActor
final class WatchWorkoutManager: NSObject, ObservableObject {
  static let shared = WatchWorkoutManager()

  @Published var routines: [WatchWorkoutPlan] = []
  @Published var activePlan: WatchWorkoutPlan?
  @Published var catalog: [WatchCatalogExercise] = []
  @Published var defaultRestSec = 90
  @Published var selectedExerciseIndex = 0
  @Published var selectedSetIndex = 0
  @Published var running = false
  @Published var paused = false
  @Published var heartRate = 0.0
  @Published var activeCalories = 0.0
  @Published var healthStartedAt: Date?
  @Published var errorMessage: String?
  @Published var timedSet: WatchCountdown?
  @Published var restTimer: WatchCountdown?
  @Published var lastSummary: WatchWorkoutSummary?
  @Published var phoneReachable = false
  @Published var lastPhoneSyncAt: Date?

  private let healthStore = HKHealthStore()
  private let encoder = JSONEncoder()
  private let decoder = JSONDecoder()
  private let iso = ISO8601DateFormatter()
  private let stateKey = "vigorly.watch.state.v2"
  private let lastPhoneSyncKey = "vigorly.watch.last-phone-sync.v1"
  private var workoutSession: HKWorkoutSession?
  private var builder: HKLiveWorkoutBuilder?
  private var timer: Timer?
  private var draftSyncTask: Task<Void, Never>?
  private var discarding = false

  override private init() {
    super.init()
    restoreState()
  }

  var currentExercise: WatchExercise? {
    guard let exercises = activePlan?.exercises,
          exercises.indices.contains(selectedExerciseIndex) else { return nil }
    return exercises[selectedExerciseIndex]
  }

  var currentSet: WatchSet? {
    guard let sets = currentExercise?.sets, sets.indices.contains(selectedSetIndex) else { return nil }
    return sets[selectedSetIndex]
  }

  var completedSets: Int {
    activePlan?.exercises.reduce(0) { total, exercise in
      total + exercise.sets.filter(\.completed).count
    } ?? 0
  }

  var totalSets: Int {
    activePlan?.exercises.reduce(0) { $0 + $1.sets.count } ?? 0
  }

  var elapsedSec: Int {
    guard let started = activePlan?.startedAt.flatMap({ iso.date(from: $0) }) ?? healthStartedAt else { return 0 }
    return max(0, Int(Date().timeIntervalSince(started)))
  }

  nonisolated func configureConnectivity() {
    guard WCSession.isSupported() else { return }
    let session = WCSession.default
    session.delegate = self
    session.activate()
    let reachable = session.isReachable
    Task { @MainActor in self.phoneReachable = reachable }
  }

  func startRoutine(_ template: WatchWorkoutPlan) {
    guard activePlan == nil else { return }
    let now = iso.string(from: Date())
    let plan = WatchWorkoutPlan(
      id: UUID().uuidString,
      name: template.name,
      routineId: template.routineId ?? template.id,
      startedAt: now,
      endedAt: nil,
      updatedAt: now,
      revision: 1,
      exercises: template.exercises.map { exercise in
        WatchExercise(
          id: UUID().uuidString,
          exerciseId: exercise.exerciseId,
          name: exercise.name,
          kind: exercise.kind,
          note: exercise.note,
          restSec: exercise.restSec,
          sets: exercise.sets.enumerated().map { index, set in
            WatchSet(
              id: UUID().uuidString,
              index: index,
              type: set.type,
              weightKg: set.weightKg ?? set.previousWeightKg,
              reps: set.reps ?? set.previousReps,
              durationSec: set.durationSec ?? set.previousDurationSec,
              previousWeightKg: set.previousWeightKg,
              previousReps: set.previousReps,
              previousDurationSec: set.previousDurationSec,
              completed: false
            )
          }
        )
      },
      heartRateBpm: nil,
      activeCalories: nil
    )
    activePlan = plan
    selectedExerciseIndex = 0
    selectedSetIndex = 0
    lastSummary = nil
    persistState()
    sendSessionEvent(type: "sessionStarted")
    Task { await startStrengthWorkout() }
  }

  func startEmptyWorkout() {
    guard activePlan == nil else { return }
    let now = iso.string(from: Date())
    activePlan = WatchWorkoutPlan(
      id: UUID().uuidString,
      name: "Workout",
      routineId: nil,
      startedAt: now,
      endedAt: nil,
      updatedAt: now,
      revision: 1,
      exercises: [],
      heartRateBpm: nil,
      activeCalories: nil
    )
    selectedExerciseIndex = 0
    selectedSetIndex = 0
    lastSummary = nil
    persistState()
    sendSessionEvent(type: "sessionStarted")
    Task { await startStrengthWorkout() }
  }

  func addExercise(_ item: WatchCatalogExercise) {
    guard var plan = activePlan else { return }
    let duration = item.kind == "duration" ? 30 : nil
    let reps = item.kind == "duration" ? nil : 10
    let exercise = WatchExercise(
      id: UUID().uuidString,
      exerciseId: item.id,
      name: item.name,
      kind: item.kind,
      note: nil,
      restSec: defaultRestSec,
      sets: (0..<3).map { index in
        WatchSet(
          id: UUID().uuidString,
          index: index,
          type: "normal",
          weightKg: nil,
          reps: reps,
          durationSec: duration,
          previousWeightKg: nil,
          previousReps: nil,
          previousDurationSec: nil,
          completed: false
        )
      }
    )
    plan.exercises.append(exercise)
    bump(&plan)
    activePlan = plan
    selectedExerciseIndex = plan.exercises.count - 1
    selectedSetIndex = 0
    persistState()
    sendSessionEvent(type: "sessionUpdated")
  }

  func removeCurrentExercise() {
    guard var plan = activePlan, plan.exercises.indices.contains(selectedExerciseIndex) else { return }
    let removedId = plan.exercises[selectedExerciseIndex].id
    plan.exercises.remove(at: selectedExerciseIndex)
    bump(&plan)
    activePlan = plan
    selectedExerciseIndex = min(selectedExerciseIndex, max(0, plan.exercises.count - 1))
    selectedSetIndex = 0
    persistState()
    sendSessionEvent(type: "sessionUpdated")
    if timedSet?.exerciseId == removedId {
      timedSet = nil
      syncTimerState()
    }
  }

  func resumeActiveWorkout() {
    guard activePlan != nil else { return }
    Task { await startStrengthWorkout() }
  }

  func startStrengthWorkout() async {
    guard let plan = activePlan else { return }
    let configuration = HKWorkoutConfiguration()
    let names = plan.exercises.map(\.name).joined(separator: " ").lowercased()
    let cardioTerms = ["treadmill", "running", "cycling", "stationary bike", "spin", "elliptical", "cross trainer", "rowing", "rower", "stair", "stepmill", "walking"]
    let hasCardio = cardioTerms.contains(where: names.contains)
    let hasStrength = plan.exercises.contains { exercise in
      !cardioTerms.contains(where: exercise.name.lowercased().contains)
    }
    if hasCardio && hasStrength {
      configuration.activityType = .crossTraining
    } else if names.contains("treadmill") || names.contains("running") {
      configuration.activityType = .running
    } else if names.contains("cycling") || names.contains("stationary bike") || names.contains("spin") {
      configuration.activityType = .cycling
    } else if names.contains("elliptical") || names.contains("cross trainer") {
      configuration.activityType = .elliptical
    } else if names.contains("rowing") || names.contains("rower") {
      configuration.activityType = .rowing
    } else if names.contains("stair") || names.contains("stepmill") {
      configuration.activityType = .stairClimbing
    } else if names.contains("walking") {
      configuration.activityType = .walking
    } else if names.contains("yoga") {
      configuration.activityType = .yoga
    } else {
      configuration.activityType = .traditionalStrengthTraining
    }
    configuration.locationType = .indoor
    await start(configuration: configuration)
  }

  func start(configuration: HKWorkoutConfiguration) async {
    guard !running else { return }
    // The iPhone launches the Watch workout immediately after publishing its
    // session. WatchConnectivity delivery can trail HealthKit by a moment, so
    // keep live tracking alive with a temporary session until that snapshot
    // arrives instead of silently dropping the launch.
    if activePlan == nil {
      let now = iso.string(from: Date())
      activePlan = WatchWorkoutPlan(
        id: UUID().uuidString,
        name: "Workout",
        routineId: nil,
        startedAt: now,
        endedAt: nil,
        updatedAt: now,
        revision: 0,
        exercises: [],
        heartRateBpm: nil,
        activeCalories: nil
      )
      persistState()
    }
    do {
      let heart = HKQuantityType(.heartRate)
      let energy = HKQuantityType(.activeEnergyBurned)
      let workout = HKObjectType.workoutType()
      try await healthStore.requestAuthorization(toShare: [workout, energy], read: [heart, energy])

      let session = try HKWorkoutSession(healthStore: healthStore, configuration: configuration)
      let builder = session.associatedWorkoutBuilder()
      builder.dataSource = HKLiveWorkoutDataSource(
        healthStore: healthStore,
        workoutConfiguration: configuration
      )
      session.delegate = self
      builder.delegate = self
      workoutSession = session
      self.builder = builder

      let start = Date()
      session.startActivity(with: start)
      try await builder.beginCollection(at: start)
      try? await session.startMirroringToCompanionDevice()
      healthStartedAt = start
      running = true
      paused = false
      errorMessage = nil
      startClock()
      sendMetrics()
    } catch {
      errorMessage = error.localizedDescription
    }
  }

  func select(exerciseIndex: Int, setIndex: Int? = nil) {
    guard let plan = activePlan, plan.exercises.indices.contains(exerciseIndex) else { return }
    selectedExerciseIndex = exerciseIndex
    if let setIndex, plan.exercises[exerciseIndex].sets.indices.contains(setIndex) {
      selectedSetIndex = setIndex
    } else {
      selectedSetIndex = plan.exercises[exerciseIndex].sets.firstIndex(where: { !$0.completed }) ?? 0
    }
  }

  func updateCurrentSet(weightKg: Double? = nil, reps: Int? = nil, durationSec: Int? = nil) {
    guard var plan = activePlan,
          plan.exercises.indices.contains(selectedExerciseIndex),
          plan.exercises[selectedExerciseIndex].sets.indices.contains(selectedSetIndex) else { return }
    if let weightKg {
      plan.exercises[selectedExerciseIndex].sets[selectedSetIndex].weightKg = min(500, max(0, weightKg))
    }
    if let reps {
      plan.exercises[selectedExerciseIndex].sets[selectedSetIndex].reps = min(999, max(0, reps))
    }
    if let durationSec {
      plan.exercises[selectedExerciseIndex].sets[selectedSetIndex].durationSec = min(3_600, max(1, durationSec))
    }
    bump(&plan)
    activePlan = plan
    scheduleDraftSync()
  }

  func completeCurrentSet(weightKg: Double?, reps: Int?, durationSec: Int? = nil) {
    guard var plan = activePlan,
          plan.exercises.indices.contains(selectedExerciseIndex),
          plan.exercises[selectedExerciseIndex].sets.indices.contains(selectedSetIndex) else { return }
    let kind = plan.exercises[selectedExerciseIndex].kind
    if kind != "duration", (reps ?? 0) <= 0 {
      errorMessage = "Choose at least 1 rep before completing this set."
      WKInterfaceDevice.current().play(.failure)
      return
    }
    var set = plan.exercises[selectedExerciseIndex].sets[selectedSetIndex]
    set.weightKg = weightKg
    set.reps = reps
    if let durationSec { set.durationSec = max(1, durationSec) }
    set.completed = true
    plan.exercises[selectedExerciseIndex].sets[selectedSetIndex] = set
    bump(&plan)
    activePlan = plan
    errorMessage = nil
    timedSet = nil
    draftSyncTask?.cancel()
    draftSyncTask = nil
    WKInterfaceDevice.current().play(.success)
    sendSessionEvent(type: "sessionUpdated")

    let rest = plan.exercises[selectedExerciseIndex].restSec
    advanceToNextSet(in: plan)
    let hasMoreSets = plan.exercises.contains { exercise in
      exercise.sets.contains(where: { !$0.completed })
    }
    if rest > 0 && hasMoreSets { restTimer = WatchCountdown(seconds: rest) }
    syncTimerState()
  }

  func reopenCurrentSet() {
    guard var plan = activePlan,
          plan.exercises.indices.contains(selectedExerciseIndex),
          plan.exercises[selectedExerciseIndex].sets.indices.contains(selectedSetIndex) else { return }
    plan.exercises[selectedExerciseIndex].sets[selectedSetIndex].completed = false
    plan.exercises[selectedExerciseIndex].sets[selectedSetIndex].rpe = nil
    bump(&plan)
    activePlan = plan
    errorMessage = nil
    persistState()
    sendSessionEvent(type: "sessionUpdated")
    WKInterfaceDevice.current().play(.click)
  }

  func toggleSet(exerciseIndex: Int, setIndex: Int) {
    guard var plan = activePlan,
          plan.exercises.indices.contains(exerciseIndex),
          plan.exercises[exerciseIndex].sets.indices.contains(setIndex) else { return }
    plan.exercises[exerciseIndex].sets[setIndex].completed.toggle()
    bump(&plan)
    activePlan = plan
    select(exerciseIndex: exerciseIndex, setIndex: setIndex)
    persistState()
    sendSessionEvent(type: "sessionUpdated")
  }

  func addSetToCurrentExercise() {
    guard var plan = activePlan, plan.exercises.indices.contains(selectedExerciseIndex) else { return }
    let last = plan.exercises[selectedExerciseIndex].sets.last
    let next = WatchSet(
      id: UUID().uuidString,
      index: plan.exercises[selectedExerciseIndex].sets.count,
      type: "normal",
      weightKg: last?.weightKg,
      reps: last?.reps,
      durationSec: last?.durationSec,
      previousWeightKg: last?.previousWeightKg,
      previousReps: last?.previousReps,
      previousDurationSec: last?.previousDurationSec,
      completed: false
    )
    plan.exercises[selectedExerciseIndex].sets.append(next)
    bump(&plan)
    activePlan = plan
    selectedSetIndex = next.index
    persistState()
    sendSessionEvent(type: "sessionUpdated")
  }

  func cycleCurrentSetType() {
    guard var plan = activePlan,
          plan.exercises.indices.contains(selectedExerciseIndex),
          plan.exercises[selectedExerciseIndex].sets.indices.contains(selectedSetIndex) else { return }
    let types = ["normal", "warmup", "failure", "drop"]
    let current = plan.exercises[selectedExerciseIndex].sets[selectedSetIndex].type
    let next = types[((types.firstIndex(of: current) ?? 0) + 1) % types.count]
    plan.exercises[selectedExerciseIndex].sets[selectedSetIndex].type = next
    bump(&plan)
    activePlan = plan
    persistState()
    sendSessionEvent(type: "sessionUpdated")
  }

  func startTimedSet() {
    guard let exercise = currentExercise, let set = currentSet else { return }
    timedSet = WatchCountdown(
      seconds: max(1, set.durationSec ?? set.previousDurationSec ?? 30),
      exerciseId: exercise.id,
      setId: set.id
    )
    syncTimerState()
    WKInterfaceDevice.current().play(.start)
  }

  func toggleTimedSetPause() {
    guard let timer = timedSet else { return }
    timedSet = timer.paused ? timer.resumed() : timer.stopped()
    syncTimerState()
  }

  func finishTimedSet() {
    guard let timer = timedSet else { return }
    completeTimedSet(timer, durationSec: max(1, timer.targetSec - timer.remaining))
  }

  func adjustRest(by seconds: Int) {
    guard let restTimer else { return }
    self.restTimer = restTimer.adjusted(by: seconds)
    syncTimerState()
  }

  func skipRest() {
    restTimer = nil
    syncTimerState()
    WKInterfaceDevice.current().play(.click)
  }

  func togglePause() {
    guard let workoutSession else { return }
    if paused { workoutSession.resume() } else { workoutSession.pause() }
  }

  func finishWorkout() {
    guard activePlan != nil else { return }
    discarding = false
    if let workoutSession { workoutSession.end() }
    else { finalizeWorkout(at: Date()) }
  }

  func discardWorkout() {
    discarding = true
    if let workoutSession { workoutSession.end() }
    else { clearActiveWorkout() }
  }

  private func advanceToNextSet(in plan: WatchWorkoutPlan) {
    for exerciseIndex in selectedExerciseIndex..<plan.exercises.count {
      let start = exerciseIndex == selectedExerciseIndex ? selectedSetIndex + 1 : 0
      if start < plan.exercises[exerciseIndex].sets.count,
         let setIndex = plan.exercises[exerciseIndex].sets.indices
          .dropFirst(start)
          .first(where: { !plan.exercises[exerciseIndex].sets[$0].completed }) {
        selectedExerciseIndex = exerciseIndex
        selectedSetIndex = setIndex
        return
      }
    }
    for exerciseIndex in plan.exercises.indices {
      if let setIndex = plan.exercises[exerciseIndex].sets.firstIndex(where: { !$0.completed }) {
        selectedExerciseIndex = exerciseIndex
        selectedSetIndex = setIndex
        return
      }
    }
  }

  private func bump(_ plan: inout WatchWorkoutPlan) {
    plan.revision += 1
    plan.updatedAt = iso.string(from: Date())
    persistState(planOverride: plan)
  }

  private func scheduleDraftSync() {
    draftSyncTask?.cancel()
    draftSyncTask = Task { [weak self] in
      try? await Task.sleep(nanoseconds: 250_000_000)
      guard !Task.isCancelled else { return }
      self?.sendSessionEvent(type: "sessionUpdated")
    }
  }

  private func startClock() {
    timer?.invalidate()
    timer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { [weak self] _ in
      Task { @MainActor in self?.tick() }
    }
  }

  private func tick() {
    if let timedSet, !timedSet.paused, timedSet.remaining <= 0 {
      completeTimedSet(timedSet, durationSec: timedSet.targetSec)
    }
    if let restTimer, restTimer.remaining <= 0 {
      self.restTimer = nil
      syncTimerState()
      WKInterfaceDevice.current().play(.notification)
    }
    objectWillChange.send()
  }

  private func completeTimedSet(_ timer: WatchCountdown, durationSec: Int) {
    guard let exerciseId = timer.exerciseId,
          let setId = timer.setId,
          let plan = activePlan,
          let exerciseIndex = plan.exercises.firstIndex(where: { $0.id == exerciseId }),
          let setIndex = plan.exercises[exerciseIndex].sets.firstIndex(where: { $0.id == setId }) else {
      timedSet = nil
      errorMessage = "That timed set is no longer in this workout."
      return
    }
    selectedExerciseIndex = exerciseIndex
    selectedSetIndex = setIndex
    let set = plan.exercises[exerciseIndex].sets[setIndex]
    completeCurrentSet(weightKg: set.weightKg, reps: set.reps, durationSec: durationSec)
  }

  private func syncTimerState() {
    guard var plan = activePlan else { return }
    plan.activeSetTimer = timedSet.flatMap { timer in
      guard let exerciseId = timer.exerciseId, let setId = timer.setId else { return nil }
      return WatchSetTimerState(
        sessionExerciseId: exerciseId,
        setId: setId,
        targetSec: timer.targetSec,
        startedAt: iso.string(from: timer.startedAt),
        endsAt: iso.string(from: timer.endsAt),
        pausedRemainingSec: timer.pausedRemaining
      )
    }
    plan.activeRestTimer = restTimer.map { timer in
      WatchRestTimerState(
        endsAt: timer.endsAt.timeIntervalSince1970 * 1_000,
        totalSec: timer.targetSec
      )
    }
    bump(&plan)
    activePlan = plan
    sendSessionEvent(type: "sessionUpdated")
  }

  private func restoreTimers(from plan: WatchWorkoutPlan) {
    if let state = plan.activeSetTimer,
       let endsAt = iso.date(from: state.endsAt) {
      timedSet = WatchCountdown(
        targetSec: state.targetSec,
        startedAt: iso.date(from: state.startedAt)
          ?? endsAt.addingTimeInterval(TimeInterval(-state.targetSec)),
        endsAt: endsAt,
        pausedRemaining: state.pausedRemainingSec,
        exerciseId: state.sessionExerciseId,
        setId: state.setId
      )
    } else {
      timedSet = nil
    }
    if let state = plan.activeRestTimer {
      let endsAt = Date(timeIntervalSince1970: state.endsAt / 1_000)
      restTimer = WatchCountdown(
        targetSec: max(1, state.totalSec),
        startedAt: endsAt.addingTimeInterval(TimeInterval(-state.totalSec)),
        endsAt: endsAt,
        pausedRemaining: nil,
        exerciseId: nil,
        setId: nil
      )
    } else {
      restTimer = nil
    }
  }

  private func finalizeWorkout(at date: Date) {
    guard var plan = activePlan else { return }
    plan.endedAt = iso.string(from: date)
    plan.heartRateBpm = heartRate > 0 ? heartRate : nil
    plan.activeCalories = activeCalories > 0 ? activeCalories : nil
    bump(&plan)
    activePlan = plan
    lastSummary = WatchWorkoutSummary(
      name: plan.name,
      elapsedSec: elapsedSec,
      completedSets: completedSets,
      activeCalories: Int(activeCalories.rounded())
    )
    sendSessionEvent(type: "sessionFinished")
    clearActiveWorkout(keepSummary: true)
  }

  private func clearActiveWorkout(keepSummary: Bool = false) {
    activePlan = nil
    selectedExerciseIndex = 0
    selectedSetIndex = 0
    running = false
    paused = false
    heartRate = 0
    activeCalories = 0
    healthStartedAt = nil
    workoutSession = nil
    builder = nil
    timedSet = nil
    restTimer = nil
    draftSyncTask?.cancel()
    draftSyncTask = nil
    timer?.invalidate()
    timer = nil
    if !keepSummary { lastSummary = nil }
    persistState()
  }

  private func sendSessionEvent(type: String) {
    guard let plan = activePlan,
          let planData = try? encoder.encode(plan),
          let planObject = try? JSONSerialization.jsonObject(with: planData) as? [String: Any] else { return }
    sendEvent([
      "type": type,
      "eventId": UUID().uuidString,
      "session": planObject,
    ])
  }

  private func sendEvent(_ event: [String: Any]) {
    guard WCSession.isSupported() else { return }
    let session = WCSession.default
    let envelope: [String: Any] = ["event": event]
    try? session.updateApplicationContext(envelope)
    if session.isReachable {
      session.sendMessage(envelope, replyHandler: nil) { error in
        session.transferUserInfo(envelope)
      }
    } else {
      session.transferUserInfo(envelope)
    }
  }

  private func sendMetrics() {
    let payload: [String: Any] = [
      "type": "metrics",
      "heartRateBpm": heartRate,
      "activeCalories": activeCalories,
      "source": "Apple Watch",
      "updatedAt": iso.string(from: Date()),
    ]
    if let workoutSession,
       let data = try? JSONSerialization.data(withJSONObject: payload) {
      Task { try? await workoutSession.sendToRemoteWorkoutSession(data: data) }
    }
    if WCSession.default.isReachable {
      WCSession.default.sendMessage(payload, replyHandler: nil)
    }
  }

  private func persistState(planOverride: WatchWorkoutPlan? = nil) {
    let state = PersistedWatchState(
      routines: routines,
      activePlan: planOverride ?? activePlan,
      catalog: catalog,
      defaultRestSec: defaultRestSec
    )
    if let data = try? encoder.encode(state) { UserDefaults.standard.set(data, forKey: stateKey) }
  }

  private func restoreState() {
    if let timestamp = UserDefaults.standard.string(forKey: lastPhoneSyncKey) {
      lastPhoneSyncAt = iso.date(from: timestamp)
    }
    guard let data = UserDefaults.standard.data(forKey: stateKey),
          let state = try? decoder.decode(PersistedWatchState.self, from: data) else { return }
    routines = state.routines
    activePlan = state.activePlan
    catalog = state.catalog
    defaultRestSec = state.defaultRestSec
    if let activePlan { restoreTimers(from: activePlan) }
  }

  private func loadLibrary(_ dictionary: [String: Any]) {
    guard JSONSerialization.isValidJSONObject(dictionary),
          let data = try? JSONSerialization.data(withJSONObject: dictionary),
          let library = try? decoder.decode(WatchWorkoutLibrary.self, from: data) else { return }
    routines = library.routines
    catalog = library.catalog
    defaultRestSec = library.defaultRestSec
    if let incoming = library.activeWorkout {
      if shouldAccept(incoming, over: activePlan) {
        activePlan = incoming
        restoreTimers(from: incoming)
        advanceToNextSet(in: incoming)
      } else if let current = activePlan, shouldAccept(current, over: incoming) {
        sendSessionEvent(type: "sessionUpdated")
      }
    } else if let current = activePlan {
      let clearedAt = library.activeWorkoutClearedAt.flatMap(iso.date(from:))
      let currentUpdatedAt = iso.date(from: current.updatedAt)
      if library.activeWorkoutClearedId == current.id,
         let clearedAt, let currentUpdatedAt, clearedAt >= currentUpdatedAt {
        discardWorkout()
      } else {
        sendSessionEvent(type: "sessionUpdated")
      }
    }
    persistState()
    let receivedAt = Date()
    lastPhoneSyncAt = receivedAt
    phoneReachable = WCSession.default.isReachable
    let timestamp = iso.string(from: receivedAt)
    UserDefaults.standard.set(timestamp, forKey: lastPhoneSyncKey)
    sendEvent([
      "type": "syncReceipt",
      "updatedAt": timestamp,
      "reachable": phoneReachable,
    ])
  }

  private func shouldAccept(_ incoming: WatchWorkoutPlan, over current: WatchWorkoutPlan?) -> Bool {
    guard let current else { return true }
    if incoming.id != current.id {
      // HealthKit can wake the Watch a moment before WatchConnectivity delivers
      // the phone's session. Replace that temporary empty workout even though
      // sensor tracking has already started.
      let temporaryHealthKitPlan = current.revision == 0
        && current.routineId == nil
        && current.exercises.isEmpty
      return temporaryHealthKitPlan || !running
    }
    if incoming.revision != current.revision { return incoming.revision > current.revision }
    guard let incomingDate = iso.date(from: incoming.updatedAt),
          let currentDate = iso.date(from: current.updatedAt) else { return false }
    return incomingDate > currentDate
  }
}

private struct PersistedWatchState: Codable {
  let routines: [WatchWorkoutPlan]
  let activePlan: WatchWorkoutPlan?
  let catalog: [WatchCatalogExercise]
  let defaultRestSec: Int
}

extension WatchWorkoutManager: HKWorkoutSessionDelegate, HKLiveWorkoutBuilderDelegate {
  nonisolated func workoutSession(
    _ workoutSession: HKWorkoutSession,
    didChangeTo toState: HKWorkoutSessionState,
    from fromState: HKWorkoutSessionState,
    date: Date
  ) {
    Task { @MainActor in
      paused = toState == .paused
      if toState == .ended {
        if let builder {
          do {
            try await builder.endCollection(at: date)
            _ = try await builder.finishWorkout()
            if let statistics = builder.statistics(for: HKQuantityType(.activeEnergyBurned)),
               let quantity = statistics.sumQuantity() {
              activeCalories = quantity.doubleValue(for: .kilocalorie())
            }
            if let statistics = builder.statistics(for: HKQuantityType(.heartRate)),
               let quantity = statistics.mostRecentQuantity() {
              heartRate = quantity.doubleValue(for: HKUnit.count().unitDivided(by: .minute()))
            }
            sendMetrics()
            try? await workoutSession.stopMirroringToCompanionDevice()
          } catch { errorMessage = error.localizedDescription }
        }
        if discarding { clearActiveWorkout() }
        else { finalizeWorkout(at: date) }
      }
    }
  }

  nonisolated func workoutSession(_ workoutSession: HKWorkoutSession, didFailWithError error: Error) {
    Task { @MainActor in errorMessage = error.localizedDescription }
  }

  nonisolated func workoutSession(
    _ workoutSession: HKWorkoutSession,
    didReceiveDataFromRemoteWorkoutSession data: [Data]
  ) {
    for packet in data {
      guard let object = try? JSONSerialization.jsonObject(with: packet) as? [String: String],
            let command = object["command"] else { continue }
      Task { @MainActor in
        if command == "pause" { workoutSession.pause() }
        if command == "resume" { workoutSession.resume() }
        if command == "end" { workoutSession.end() }
        if command == "discard" { discardWorkout() }
      }
    }
  }

  nonisolated func workoutBuilderDidCollectEvent(_ workoutBuilder: HKLiveWorkoutBuilder) {}

  nonisolated func workoutBuilder(
    _ workoutBuilder: HKLiveWorkoutBuilder,
    didCollectDataOf collectedTypes: Set<HKSampleType>
  ) {
    Task { @MainActor in
      if collectedTypes.contains(HKQuantityType(.heartRate)),
         let statistics = workoutBuilder.statistics(for: HKQuantityType(.heartRate)),
         let quantity = statistics.mostRecentQuantity() {
        heartRate = quantity.doubleValue(for: HKUnit.count().unitDivided(by: .minute()))
      }
      if collectedTypes.contains(HKQuantityType(.activeEnergyBurned)),
         let statistics = workoutBuilder.statistics(for: HKQuantityType(.activeEnergyBurned)),
         let quantity = statistics.sumQuantity() {
        activeCalories = quantity.doubleValue(for: .kilocalorie())
      }
      sendMetrics()
    }
  }
}

extension WatchWorkoutManager: WCSessionDelegate {
  nonisolated func session(
    _ session: WCSession,
    activationDidCompleteWith activationState: WCSessionActivationState,
    error: Error?
  ) {
    let library = session.receivedApplicationContext["library"] as? [String: Any]
    let reachable = session.isReachable
    Task { @MainActor in
      phoneReachable = reachable
      if let library { loadLibrary(library) }
      if let error { errorMessage = error.localizedDescription }
    }
  }

  nonisolated func session(_ session: WCSession, didReceiveApplicationContext applicationContext: [String: Any]) {
    guard let library = applicationContext["library"] as? [String: Any] else { return }
    Task { @MainActor in loadLibrary(library) }
  }

  nonisolated func session(_ session: WCSession, didReceiveMessage message: [String: Any]) {
    Task { @MainActor in receiveConnectivity(message) }
  }

  nonisolated func session(_ session: WCSession, didReceiveUserInfo userInfo: [String: Any] = [:]) {
    Task { @MainActor in receiveConnectivity(userInfo) }
  }

  nonisolated func sessionReachabilityDidChange(_ session: WCSession) {
    let reachable = session.isReachable
    Task { @MainActor in
      phoneReachable = reachable
      if reachable, activePlan != nil { sendSessionEvent(type: "sessionUpdated") }
    }
  }

  private func receiveConnectivity(_ message: [String: Any]) {
    if let library = message["library"] as? [String: Any] {
      loadLibrary(library)
      return
    }
    guard let command = message["command"] as? String else { return }
    if command == "pause" { workoutSession?.pause() }
    if command == "resume" { workoutSession?.resume() }
    if command == "end" { finishWorkout() }
    if command == "discard" { discardWorkout() }
  }
}
