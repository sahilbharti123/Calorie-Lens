import Foundation
import HealthKit
import React
import WatchConnectivity

@objc(VigorlyWorkoutBridge)
final class VigorlyWorkoutBridge: RCTEventEmitter {
  private var observer: NSObjectProtocol?

  override init() {
    super.init()
    observer = NotificationCenter.default.addObserver(
      forName: .vigorlyWorkoutEvent,
      object: nil,
      queue: .main
    ) { [weak self] notification in
      guard let payload = notification.userInfo else { return }
      self?.sendEvent(withName: "VigorlyWorkoutEvent", body: payload)
    }
  }

  deinit {
    if let observer { NotificationCenter.default.removeObserver(observer) }
  }

  @objc override static func requiresMainQueueSetup() -> Bool { true }
  override func supportedEvents() -> [String]! { ["VigorlyWorkoutEvent"] }

  @objc(syncWorkout:)
  func syncWorkout(_ payload: NSDictionary) {
    guard #available(iOS 17.0, *) else { return }
    VigorlyWorkoutManager.shared.syncWorkout(payload)
  }

  @objc(sendCommand:)
  func sendCommand(_ command: String) {
    guard #available(iOS 17.0, *) else { return }
    VigorlyWorkoutManager.shared.send(command: command)
  }

  @objc(drainEvents:rejecter:)
  func drainEvents(
    _ resolve: RCTPromiseResolveBlock,
    rejecter reject: RCTPromiseRejectBlock
  ) {
    guard #available(iOS 17.0, *) else {
      resolve([])
      return
    }
    resolve(VigorlyWorkoutManager.shared.drainEvents())
  }
}

extension Notification.Name {
  static let vigorlyWorkoutEvent = Notification.Name("VigorlyWorkoutEvent")
}

@available(iOS 17.0, *)
final class VigorlyWorkoutManager: NSObject, HKWorkoutSessionDelegate, WCSessionDelegate {
  static let shared = VigorlyWorkoutManager()

  private let healthStore = HKHealthStore()
  private var mirroredSession: HKWorkoutSession?
  private let pendingEventsKey = "vigorly.watch.pending-events.v2"
  private var pendingEvents: [[String: Any]] = []

  override private init() {
    if let data = UserDefaults.standard.data(forKey: pendingEventsKey),
       let saved = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] {
      pendingEvents = saved
    }
    super.init()
  }

  func configure() {
    healthStore.workoutSessionMirroringStartHandler = { [weak self] session in
      self?.attach(session)
    }
    guard WCSession.isSupported() else { return }
    let connectivity = WCSession.default
    connectivity.delegate = self
    connectivity.activate()
  }

  private func attach(_ session: HKWorkoutSession) {
    mirroredSession = session
    session.delegate = self
    emit(["type": "state", "state": "connected", "source": "Apple Watch"])
  }

  func syncWorkout(_ payload: NSDictionary) {
    guard WCSession.isSupported() else { return }
    let data = payload as? [String: Any] ?? [:]
    do {
      let message: [String: Any] = ["library": data]
      try WCSession.default.updateApplicationContext(message)
      if WCSession.default.isReachable {
        WCSession.default.sendMessage(message, replyHandler: nil) { [weak self] error in
          self?.emit(["type": "error", "message": error.localizedDescription])
        }
      }
    } catch {
      emit(["type": "error", "message": error.localizedDescription])
    }
  }

  func send(command: String) {
    let payload = ["command": command]
    if let session = mirroredSession,
       let data = try? JSONSerialization.data(withJSONObject: payload) {
      Task {
        do { try await session.sendToRemoteWorkoutSession(data: data) }
        catch { emit(["type": "error", "message": error.localizedDescription]) }
      }
    }
    guard WCSession.isSupported() else { return }
    let connectivity = WCSession.default
    if connectivity.isReachable {
      connectivity.sendMessage(payload, replyHandler: nil) { _ in
        connectivity.transferUserInfo(payload)
      }
    } else {
      connectivity.transferUserInfo(payload)
    }
  }

  func drainEvents() -> [[String: Any]] {
    defer {
      pendingEvents.removeAll()
      persistPendingEvents()
    }
    return pendingEvents
  }

  private func emit(_ payload: [String: Any]) {
    pendingEvents.append(payload)
    if pendingEvents.count > 100 { pendingEvents.removeFirst(pendingEvents.count - 100) }
    persistPendingEvents()
    NotificationCenter.default.post(name: .vigorlyWorkoutEvent, object: nil, userInfo: payload)
  }

  private func persistPendingEvents() {
    guard JSONSerialization.isValidJSONObject(pendingEvents),
          let data = try? JSONSerialization.data(withJSONObject: pendingEvents) else { return }
    UserDefaults.standard.set(data, forKey: pendingEventsKey)
  }

  private func receive(_ message: [String: Any]) {
    if let event = message["event"] as? [String: Any] { emit(event) }
    else { emit(message) }
  }

  func workoutSession(
    _ workoutSession: HKWorkoutSession,
    didChangeTo toState: HKWorkoutSessionState,
    from fromState: HKWorkoutSessionState,
    date: Date
  ) {
    emit(["type": "state", "state": toState.rawValue, "source": "Apple Watch"])
  }

  func workoutSession(_ workoutSession: HKWorkoutSession, didFailWithError error: Error) {
    emit(["type": "error", "message": error.localizedDescription])
  }

  func workoutSession(
    _ workoutSession: HKWorkoutSession,
    didReceiveDataFromRemoteWorkoutSession data: [Data]
  ) {
    for packet in data {
      guard let payload = try? JSONSerialization.jsonObject(with: packet) as? [String: Any] else { continue }
      emit(payload)
    }
  }

  func session(
    _ session: WCSession,
    activationDidCompleteWith activationState: WCSessionActivationState,
    error: Error?
  ) {
    if let error { emit(["type": "error", "message": error.localizedDescription]) }
  }

  func sessionDidBecomeInactive(_ session: WCSession) {}
  func sessionDidDeactivate(_ session: WCSession) { session.activate() }

  func session(_ session: WCSession, didReceiveMessage message: [String: Any]) {
    receive(message)
  }

  func session(
    _ session: WCSession,
    didReceiveMessage message: [String: Any],
    replyHandler: @escaping ([String: Any]) -> Void
  ) {
    receive(message)
    replyHandler(["received": true])
  }

  func session(_ session: WCSession, didReceiveUserInfo userInfo: [String: Any] = [:]) {
    receive(userInfo)
  }

  func session(_ session: WCSession, didReceiveApplicationContext applicationContext: [String: Any]) {
    receive(applicationContext)
  }
}
