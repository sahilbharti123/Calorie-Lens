import XCTest

final class VigorlyUITests: XCTestCase {
  override func setUpWithError() throws {
    continueAfterFailure = false
  }

  @MainActor
  func testAuthSurfaceKeepsCoreControlsReachable() throws {
    let app = XCUIApplication()
    app.launch()

    XCTAssertTrue(app.staticTexts["Your progress, on every device."].waitForExistence(timeout: 12))
    XCTAssertTrue(app.textFields["Email"].exists)
    XCTAssertTrue(app.secureTextFields["Password"].exists)

    let signIn = app.buttons["Sign in"]
    if !signIn.isHittable { app.swipeUp() }
    XCTAssertTrue(signIn.waitForExistence(timeout: 3))

    let guest = app.buttons["Use guest mode on this device"]
    if !guest.isHittable { app.swipeUp() }
    XCTAssertTrue(guest.waitForExistence(timeout: 3))
  }

  @MainActor
  func testCreateAccountFieldsAndKeyboardDismissal() throws {
    let app = XCUIApplication()
    app.launch()

    let createAccount = app.descendants(matching: .any)["Create account"]
    XCTAssertTrue(createAccount.waitForExistence(timeout: 12))
    createAccount.tap()

    let name = app.textFields["Your name"]
    let email = app.textFields["Email"]
    let password = app.secureTextFields["Password"]
    XCTAssertTrue(name.waitForExistence(timeout: 3))
    XCTAssertTrue(email.exists)
    XCTAssertTrue(password.exists)

    name.tap()
    name.typeText("QA User")
    let keyboard = app.keyboards.firstMatch
    XCTAssertTrue(keyboard.exists)
    let form = app.descendants(matching: .any)["auth-form-scroll"]
    XCTAssertTrue(form.exists)
    form.swipeUp()
    XCTAssertTrue(keyboard.waitForNonExistence(timeout: 3))
  }
}
