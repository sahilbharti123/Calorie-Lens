import SwiftUI

struct WorkoutView: View {
  @EnvironmentObject private var workout: WatchWorkoutManager

  var body: some View {
    Group {
      if workout.activePlan != nil {
        LiveWorkoutView()
      } else if let summary = workout.lastSummary {
        WorkoutSummaryView(summary: summary)
      } else {
        RoutineLibraryView()
      }
    }
    .tint(.vigorLime)
  }
}

private struct RoutineLibraryView: View {
  @EnvironmentObject private var workout: WatchWorkoutManager

  var body: some View {
    ScrollView {
      VStack(alignment: .leading, spacing: 10) {
        HStack(spacing: 7) {
          Image(systemName: "figure.strengthtraining.traditional")
            .font(.system(size: 17, weight: .bold))
            .foregroundStyle(Color.vigorLime)
          Text("Vigorly")
            .font(.system(size: 18, weight: .bold, design: .rounded))
        }

        WatchSyncStatusRow()

        Button {
          workout.startEmptyWorkout()
        } label: {
          Label("Quick workout", systemImage: "plus")
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .buttonStyle(.borderedProminent)
        .tint(.vigorLime)
        .foregroundStyle(.black)

        if workout.routines.isEmpty {
          VStack(alignment: .leading, spacing: 4) {
            Image(systemName: "list.bullet.rectangle")
              .foregroundStyle(Color.vigorLime)
            Text("No routines yet")
              .font(.headline)
            Text("Create one on iPhone, or start a quick workout here.")
              .font(.caption2)
              .foregroundStyle(.secondary)
          }
          .padding(.vertical, 8)
        } else {
          Text("ROUTINES")
            .font(.system(size: 10, weight: .bold))
            .foregroundStyle(.secondary)
          ForEach(workout.routines) { routine in
            Button {
              workout.startRoutine(routine)
            } label: {
              HStack(spacing: 9) {
                ZStack {
                  Circle().fill(Color.vigorLime.opacity(0.16))
                  Image(systemName: "play.fill")
                    .font(.system(size: 10, weight: .bold))
                    .foregroundStyle(Color.vigorLime)
                    .offset(x: 1)
                }
                .frame(width: 32, height: 32)
                VStack(alignment: .leading, spacing: 2) {
                  Text(routine.name).font(.headline).lineLimit(1)
                  Text("\(routine.exercises.count) exercises · \(setCount(routine)) sets")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                }
              }
              .frame(maxWidth: .infinity, alignment: .leading)
            }
            .buttonStyle(.plain)
            .padding(.vertical, 5)
          }
        }

        if let error = workout.errorMessage {
          Label(error, systemImage: "exclamationmark.triangle.fill")
            .font(.caption2)
            .foregroundStyle(.red)
        }
      }
      .padding(.horizontal, 8)
      .padding(.bottom, 12)
    }
  }

  private func setCount(_ routine: WatchWorkoutPlan) -> Int {
    routine.exercises.reduce(0) { $0 + $1.sets.count }
  }
}

private enum LiveScreen {
  case editor
  case overview
  case exercisePicker
  case setActions
}

private struct WatchSyncStatusRow: View {
  @EnvironmentObject private var workout: WatchWorkoutManager

  var body: some View {
    HStack(spacing: 5) {
      Circle()
        .fill(workout.phoneReachable ? Color.vigorLime : Color.secondary)
        .frame(width: 6, height: 6)
      Text(workout.phoneReachable ? "Phone connected" : "Offline ready")
        .lineLimit(1)
      Spacer(minLength: 3)
      if let lastSync = workout.lastPhoneSyncAt {
        Text(lastSyncLabel(lastSync))
          .lineLimit(1)
      }
    }
    .font(.system(size: 9, weight: .semibold))
    .foregroundStyle(.secondary)
    .accessibilityElement(children: .combine)
  }

  private func lastSyncLabel(_ date: Date) -> String {
    let elapsed = max(0, Int(Date().timeIntervalSince(date)))
    if elapsed < 60 { return "Synced now" }
    if elapsed < 3_600 { return "Synced \(elapsed / 60)m ago" }
    return "Synced \(elapsed / 3_600)h ago"
  }
}

private struct LiveWorkoutView: View {
  @EnvironmentObject private var workout: WatchWorkoutManager
  @State private var screen: LiveScreen = .editor

  var body: some View {
    Group {
      if workout.restTimer != nil {
        RestTimerView()
      } else {
        switch screen {
        case .editor:
          SetEditorView(
            showOverview: { screen = .overview },
            showExercisePicker: { screen = .exercisePicker },
            showActions: { screen = .setActions }
          )
        case .overview:
          WorkoutOverviewView(
            editCurrentSet: { screen = .editor },
            showExercisePicker: { screen = .exercisePicker }
          )
        case .exercisePicker:
          ExercisePickerView(done: { screen = .editor })
        case .setActions:
          SetActionsView(done: { screen = .editor }, removed: { screen = .overview })
        }
      }
    }
    .onAppear {
      if workout.activePlan?.exercises.isEmpty == true { screen = .overview }
    }
  }
}

private struct SetEditorView: View {
  @EnvironmentObject private var workout: WatchWorkoutManager
  let showOverview: () -> Void
  let showExercisePicker: () -> Void
  let showActions: () -> Void

  var body: some View {
    if let exercise = workout.currentExercise, let set = workout.currentSet {
      VStack(alignment: .leading, spacing: 7) {
        HStack {
          Button(action: showOverview) {
            Image(systemName: "chevron.left")
          }
          .buttonStyle(.plain)
          .accessibilityLabel("Workout overview")
          Spacer()
          Text(clock(workout.elapsedSec))
            .font(.system(size: 13, weight: .semibold, design: .rounded))
            .monospacedDigit()
            .foregroundStyle(.secondary)
          Button(action: showActions) {
            Image(systemName: "ellipsis")
          }
          .buttonStyle(.plain)
          .accessibilityLabel("Set actions")
        }

        Text(exercise.name)
          .font(.system(size: 19, weight: .bold, design: .rounded))
          .lineLimit(2)
          .minimumScaleFactor(0.82)

        HStack(spacing: 5) {
          SetTypeBadge(type: set.type)
          Text("Set \(set.index + 1) of \(exercise.sets.count)")
            .font(.caption2)
            .foregroundStyle(.secondary)
          Spacer()
          if workout.heartRate > 0 {
            Label("\(Int(workout.heartRate))", systemImage: "heart.fill")
              .font(.caption2.bold())
              .foregroundStyle(.red)
          }
        }

        if exercise.kind == "duration" {
          DurationEditor(set: set)
        } else {
          if let previous = previousLabel(set: set, kind: exercise.kind) {
            Text("Previous  \(previous)")
              .font(.system(size: 10, weight: .medium))
              .foregroundStyle(.secondary)
              .lineLimit(1)
          }
          HStack(spacing: 7) {
            if exercise.kind == "weight-reps" {
              InlineMetricPicker(
                title: "KG",
                selection: weightSelection,
                range: 0...1_000,
                valueLabel: { weightLabel(Double($0) / 2) }
              )
            }
            InlineMetricPicker(
              title: "REPS",
              selection: repsSelection,
              range: 0...999,
              valueLabel: { "\($0)" }
            )
          }
        }

        if let note = exercise.note, !note.isEmpty {
          Label(note, systemImage: "note.text")
            .font(.system(size: 10))
            .foregroundStyle(.secondary)
            .lineLimit(1)
        }

        HStack(spacing: 7) {
          Button {
            previousSet(in: exercise)
          } label: {
            Image(systemName: "arrow.left")
              .frame(maxWidth: .infinity)
          }
          .buttonStyle(.bordered)
          .accessibilityLabel("Previous set")

          Button {
            if workout.completedSets == workout.totalSets, workout.totalSets > 0 {
              workout.finishWorkout()
            } else if exercise.kind == "duration" {
              if workout.timedSet == nil { workout.startTimedSet() }
              else { workout.finishTimedSet() }
            } else {
              workout.completeCurrentSet(
                weightKg: exercise.kind == "weight-reps"
                  ? (set.weightKg ?? set.previousWeightKg ?? 0)
                  : nil,
                reps: set.reps ?? set.previousReps ?? 0
              )
            }
          } label: {
            Text(primaryLabel(kind: exercise.kind))
              .font(.headline)
              .frame(maxWidth: .infinity)
          }
          .buttonStyle(.borderedProminent)
          .tint(.vigorLime)
          .foregroundStyle(.black)
        }
        if let error = workout.errorMessage {
          Label(error, systemImage: "exclamationmark.triangle.fill")
            .font(.system(size: 9, weight: .medium))
            .foregroundStyle(.red)
            .lineLimit(2)
        }
      }
      .padding(.horizontal, 7)
    } else {
      VStack(spacing: 9) {
        Image(systemName: "dumbbell.fill")
          .font(.title2)
          .foregroundStyle(Color.vigorLime)
        Text("Add your first exercise")
          .font(.headline)
          .multilineTextAlignment(.center)
        Button("Choose exercise", action: showExercisePicker)
          .buttonStyle(.borderedProminent)
          .tint(.vigorLime)
          .foregroundStyle(.black)
        Button("Workout overview", action: showOverview)
          .font(.caption)
      }
      .padding()
    }
  }

  private var weightSelection: Binding<Int> {
    Binding(
      get: {
        min(1_000, max(0, Int(((workout.currentSet?.weightKg
          ?? workout.currentSet?.previousWeightKg
          ?? 0) * 2).rounded())))
      },
      set: { workout.updateCurrentSet(weightKg: Double($0) / 2) }
    )
  }

  private var repsSelection: Binding<Int> {
    Binding(
      get: { min(999, max(0, workout.currentSet?.reps ?? workout.currentSet?.previousReps ?? 0)) },
      set: { workout.updateCurrentSet(reps: $0) }
    )
  }

  private func weightLabel(_ weight: Double) -> String {
    weight.rounded() == weight ? "\(Int(weight))" : String(format: "%.1f", weight)
  }

  private func previousSet(in exercise: WatchExercise) {
    if workout.selectedSetIndex > 0 {
      workout.select(exerciseIndex: workout.selectedExerciseIndex, setIndex: workout.selectedSetIndex - 1)
    } else if workout.selectedExerciseIndex > 0 {
      let index = workout.selectedExerciseIndex - 1
      let setIndex = max(0, (workout.activePlan?.exercises[index].sets.count ?? 1) - 1)
      workout.select(exerciseIndex: index, setIndex: setIndex)
    }
  }

  private func primaryLabel(kind: String) -> String {
    if workout.completedSets == workout.totalSets, workout.totalSets > 0 { return "Finish" }
    if kind == "duration" { return workout.timedSet == nil ? "Start" : "Done" }
    return "Next"
  }
}

private struct InlineMetricPicker: View {
  let title: String
  @Binding var selection: Int
  let range: ClosedRange<Int>
  let valueLabel: (Int) -> String

  var body: some View {
    VStack(spacing: 1) {
      Text(title)
        .font(.system(size: 9, weight: .bold))
        .foregroundStyle(Color.vigorLime)
      Picker(title, selection: $selection) {
        ForEach(range, id: \.self) { option in
          Text(valueLabel(option))
            .font(.system(size: 20, weight: .semibold, design: .rounded))
            .monospacedDigit()
            .tag(option)
        }
      }
      .pickerStyle(.wheel)
      .labelsHidden()
      .frame(maxWidth: .infinity)
      .frame(height: 62)
      .clipped()
    }
    .frame(maxWidth: .infinity)
    .padding(.top, 4)
    .background(Color.white.opacity(0.07), in: RoundedRectangle(cornerRadius: 12))
    .overlay {
      RoundedRectangle(cornerRadius: 12)
        .stroke(Color.vigorLime.opacity(0.35), lineWidth: 1)
    }
    .accessibilityHint("Swipe vertically or turn the Digital Crown")
  }
}

private struct SetActionsView: View {
  @EnvironmentObject private var workout: WatchWorkoutManager
  let done: () -> Void
  let removed: () -> Void

  var body: some View {
    ScrollView {
      VStack(alignment: .leading, spacing: 7) {
        HStack {
          Button(action: done) { Image(systemName: "chevron.left") }
            .buttonStyle(.plain)
          Text("Set actions").font(.headline)
        }
        Button {
          workout.addSetToCurrentExercise()
          done()
        } label: {
          Label("Add set", systemImage: "plus.circle.fill")
        }
        .tint(.vigorLime)
        Button {
          workout.cycleCurrentSetType()
          done()
        } label: {
          Label("Change set type", systemImage: "arrow.triangle.2.circlepath")
        }
        if workout.currentSet?.completed == true {
          Button {
            workout.reopenCurrentSet()
            done()
          } label: {
            Label("Reopen completed set", systemImage: "arrow.uturn.backward.circle.fill")
          }
          .tint(.vigorLime)
        }
        Button(role: .destructive) {
          workout.removeCurrentExercise()
          removed()
        } label: {
          Label("Remove exercise", systemImage: "trash")
        }
      }
      .padding(.horizontal, 8)
    }
  }
}

private struct DurationEditor: View {
  @EnvironmentObject private var workout: WatchWorkoutManager
  let set: WatchSet

  var body: some View {
    VStack(spacing: 7) {
      if let timer = workout.timedSet {
        Gauge(value: Double(timer.targetSec - timer.remaining), in: 0...Double(timer.targetSec)) {
          Image(systemName: "timer")
        } currentValueLabel: {
          Text(clock(timer.remaining))
            .font(.system(size: 21, weight: .bold, design: .rounded))
            .monospacedDigit()
        }
        .gaugeStyle(.accessoryCircularCapacity)
        .tint(.vigorLime)
        HStack {
          Button {
            workout.toggleTimedSetPause()
          } label: {
            Image(systemName: timer.paused ? "play.fill" : "pause.fill")
          }
          Button {
            workout.finishTimedSet()
          } label: {
            Image(systemName: "checkmark")
          }
          .tint(.vigorLime)
        }
      } else {
        VStack(spacing: 4) {
          InlineMetricPicker(
            title: "SECONDS",
            selection: durationSelection,
            range: 5...600,
            valueLabel: clock
          )
          if let previous = set.previousDurationSec {
            Text("Previous \(clock(previous))")
              .font(.caption2)
              .foregroundStyle(.secondary)
          }
        }
      }
    }
  }

  private var durationSelection: Binding<Int> {
    Binding(
      get: { min(600, max(5, workout.currentSet?.durationSec ?? workout.currentSet?.previousDurationSec ?? 30)) },
      set: { workout.updateCurrentSet(durationSec: $0) }
    )
  }
}

private struct RestTimerView: View {
  @EnvironmentObject private var workout: WatchWorkoutManager

  var body: some View {
    if let timer = workout.restTimer {
      VStack(spacing: 8) {
        Text("REST")
          .font(.system(size: 10, weight: .bold))
          .foregroundStyle(Color.vigorLime)
        Gauge(value: Double(timer.targetSec - timer.remaining), in: 0...Double(timer.targetSec)) {
          Image(systemName: "timer")
        } currentValueLabel: {
          Text(clock(timer.remaining))
            .font(.system(size: 22, weight: .bold, design: .rounded))
            .monospacedDigit()
        }
        .gaugeStyle(.accessoryCircularCapacity)
        .tint(.vigorLime)
        if let next = workout.currentExercise, let set = workout.currentSet {
          Text("Next · \(next.name)")
            .font(.caption.bold())
            .lineLimit(1)
          Text("Set \(set.index + 1)")
            .font(.caption2)
            .foregroundStyle(.secondary)
        }
        HStack(spacing: 5) {
          Button("−15") { workout.adjustRest(by: -15) }
          Button("Skip") { workout.skipRest() }
            .tint(.vigorLime)
          Button("+15") { workout.adjustRest(by: 15) }
        }
        .font(.caption2.bold())
      }
      .padding(.horizontal, 6)
    }
  }
}

private struct WorkoutOverviewView: View {
  @EnvironmentObject private var workout: WatchWorkoutManager
  let editCurrentSet: () -> Void
  let showExercisePicker: () -> Void

  var body: some View {
    ScrollView {
      VStack(alignment: .leading, spacing: 9) {
        HStack(alignment: .firstTextBaseline) {
          VStack(alignment: .leading, spacing: 1) {
            Text(workout.activePlan?.name ?? "Workout")
              .font(.headline)
              .lineLimit(1)
            Text("\(workout.completedSets)/\(workout.totalSets) sets")
              .font(.caption2)
              .foregroundStyle(.secondary)
          }
          Spacer()
          Text(clock(workout.elapsedSec))
            .font(.caption.bold())
            .monospacedDigit()
        }

        WatchSyncStatusRow()

        if workout.running {
          HStack(spacing: 12) {
            Label(workout.heartRate > 0 ? "\(Int(workout.heartRate))" : "—", systemImage: "heart.fill")
              .foregroundStyle(.red)
            Label("\(Int(workout.activeCalories))", systemImage: "flame.fill")
              .foregroundStyle(.orange)
          }
          .font(.headline)
        } else {
          Button {
            workout.resumeActiveWorkout()
          } label: {
            Label("Start live tracking", systemImage: "applewatch")
          }
          .buttonStyle(.borderedProminent)
          .tint(.vigorLime)
          .foregroundStyle(.black)
        }

        ForEach(Array((workout.activePlan?.exercises ?? []).enumerated()), id: \.element.id) { exerciseIndex, exercise in
          VStack(alignment: .leading, spacing: 5) {
            Button {
              workout.select(exerciseIndex: exerciseIndex)
              editCurrentSet()
            } label: {
              HStack {
                Text(exercise.name).font(.headline).lineLimit(1)
                Spacer()
                Image(systemName: "chevron.right").font(.caption2).foregroundStyle(.secondary)
              }
            }
            .buttonStyle(.plain)
            HStack(spacing: 5) {
              ForEach(Array(exercise.sets.enumerated()), id: \.element.id) { setIndex, set in
                Button {
                  workout.select(exerciseIndex: exerciseIndex, setIndex: setIndex)
                  editCurrentSet()
                } label: {
                  ZStack {
                    Circle().fill(set.completed ? Color.vigorLime : Color.white.opacity(0.09))
                    if set.completed {
                      Image(systemName: "checkmark")
                        .font(.system(size: 9, weight: .bold))
                        .foregroundStyle(.black)
                    } else {
                      Text("\(setIndex + 1)").font(.caption2.bold())
                    }
                  }
                  .frame(width: 28, height: 28)
                }
                .buttonStyle(.plain)
                .accessibilityLabel("\(exercise.name), set \(setIndex + 1), \(set.completed ? "complete" : "not complete")")
              }
            }
            if let note = exercise.note, !note.isEmpty {
              Label(note, systemImage: "note.text")
                .font(.caption2)
                .foregroundStyle(.secondary)
                .lineLimit(2)
            }
          }
          .padding(.vertical, 4)
        }

        Button {
          showExercisePicker()
        } label: {
          Label("Add exercise", systemImage: "plus")
        }
        .tint(.vigorLime)

        if workout.running {
          Button(workout.paused ? "Resume workout" : "Pause workout") { workout.togglePause() }
        }
        Button("Finish workout") { workout.finishWorkout() }
          .tint(.vigorLime)
        Button("Discard", role: .destructive) { workout.discardWorkout() }
      }
      .padding(.horizontal, 8)
      .padding(.bottom, 12)
    }
  }
}

private struct ExercisePickerView: View {
  @EnvironmentObject private var workout: WatchWorkoutManager
  @State private var query = ""
  let done: () -> Void

  private var filteredCatalog: [WatchCatalogExercise] {
    let needle = query.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    guard !needle.isEmpty else { return workout.catalog }
    return workout.catalog.filter {
      $0.name.lowercased().contains(needle) || $0.primaryMuscle.lowercased().contains(needle)
    }
  }

  var body: some View {
    ScrollView {
      LazyVStack(alignment: .leading, spacing: 5) {
        HStack {
          Button(action: done) { Image(systemName: "chevron.left") }
            .buttonStyle(.plain)
          Text("Add exercise").font(.headline)
        }
        if workout.catalog.isEmpty {
          Text("Open the iPhone app once to sync the exercise library.")
            .font(.caption)
            .foregroundStyle(.secondary)
        }
        ForEach(filteredCatalog) { exercise in
          Button {
            workout.addExercise(exercise)
            done()
          } label: {
            HStack(spacing: 8) {
              Image(systemName: exercise.kind == "duration" ? "timer" : "dumbbell.fill")
                .foregroundStyle(Color.vigorLime)
                .frame(width: 22)
              VStack(alignment: .leading, spacing: 1) {
                Text(exercise.name).font(.caption.bold()).lineLimit(1)
                Text(exercise.primaryMuscle.capitalized)
                  .font(.system(size: 9))
                  .foregroundStyle(.secondary)
              }
              Spacer()
              Image(systemName: "plus.circle.fill").foregroundStyle(Color.vigorLime)
            }
            .padding(.vertical, 5)
          }
          .buttonStyle(.plain)
        }
      }
      .padding(.horizontal, 8)
    }
    .searchable(text: $query, prompt: "Exercise")
  }
}

private struct WorkoutSummaryView: View {
  @EnvironmentObject private var workout: WatchWorkoutManager
  let summary: WatchWorkoutSummary

  var body: some View {
    VStack(spacing: 9) {
      Image(systemName: "checkmark.circle.fill")
        .font(.system(size: 34))
        .foregroundStyle(Color.vigorLime)
      Text("Workout saved")
        .font(.headline)
      Text(summary.name)
        .font(.caption)
        .foregroundStyle(.secondary)
        .lineLimit(1)
      HStack(spacing: 14) {
        summaryMetric(value: clock(summary.elapsedSec), label: "TIME")
        summaryMetric(value: "\(summary.completedSets)", label: "SETS")
        summaryMetric(value: "\(summary.activeCalories)", label: "KCAL")
      }
      Button("Done") { workout.lastSummary = nil }
        .buttonStyle(.borderedProminent)
        .tint(.vigorLime)
        .foregroundStyle(.black)
    }
    .padding(.horizontal, 8)
  }

  private func summaryMetric(value: String, label: String) -> some View {
    VStack(spacing: 1) {
      Text(value).font(.caption.bold()).monospacedDigit()
      Text(label).font(.system(size: 8, weight: .bold)).foregroundStyle(.secondary)
    }
  }
}

private struct SetTypeBadge: View {
  let type: String

  var body: some View {
    if type != "normal" {
      Text(label)
        .font(.system(size: 8, weight: .bold))
        .foregroundStyle(type == "failure" ? .red : Color.vigorLime)
        .padding(.horizontal, 5)
        .padding(.vertical, 2)
        .background(Color.white.opacity(0.08), in: Capsule())
    }
  }

  private var label: String {
    switch type {
    case "warmup": "WARM-UP"
    case "failure": "FAILURE"
    case "drop": "DROP"
    default: ""
    }
  }
}

private func previousLabel(set: WatchSet, kind: String) -> String? {
  if kind == "reps-only", let reps = set.previousReps { return "\(reps) reps" }
  if kind == "weight-reps", let weight = set.previousWeightKg, let reps = set.previousReps {
    let weightText = weight.rounded() == weight ? "\(Int(weight))" : String(format: "%.1f", weight)
    return "\(weightText) kg × \(reps)"
  }
  return nil
}

private func clock(_ seconds: Int) -> String {
  let total = max(0, seconds)
  return String(format: "%02d:%02d", total / 60, total % 60)
}

private extension Color {
  static let vigorLime = Color(red: 0.78, green: 1, blue: 0.24)
}
