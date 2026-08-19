import SwiftUI
import WatchKit
import HealthKit

@main
struct VigorlyWatchApp: App {
  @WKApplicationDelegateAdaptor(WatchAppDelegate.self) private var appDelegate
  @StateObject private var workout = WatchWorkoutManager.shared

  var body: some Scene {
    WindowGroup {
      WorkoutView().environmentObject(workout)
    }
  }
}

final class WatchAppDelegate: NSObject, WKApplicationDelegate {
  func applicationDidFinishLaunching() {
    WatchWorkoutManager.shared.configureConnectivity()
  }

  func handle(_ workoutConfiguration: HKWorkoutConfiguration) {
    Task { await WatchWorkoutManager.shared.start(configuration: workoutConfiguration) }
  }
}
