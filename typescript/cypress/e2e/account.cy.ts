/// <reference types="cypress" />

describe("Account Tab", () => {
  beforeEach(() => {
    cy.visit("/");
    cy.navigateToTab("account");
  });

  describe("Layout", () => {
    it("should display account tab content", () => {
      cy.contains("button", "Account").should("have.class", "text-white");
    });

    it("should display account management section", () => {
      cy.contains(/Account|Profile|Settings/i).should("be.visible");
    });
  });

  describe("Not Logged In State", () => {
    it("should prompt user to login", () => {
      cy.contains(/Login|Sign in/i).should("be.visible");
    });
  });

  describe("Logged In State", () => {
    beforeEach(() => {
      // Simulate logged in state by first logging in
      cy.mockTauriInvoke("login", {
        success: true,
        message: "Login successful",
        username: "testuser",
      });

      cy.contains("button", "Login").click();
      cy.get('input[placeholder="Enter username"]').type("testuser");
      cy.get('input[placeholder="Enter password"]').type("password123");
      cy.get('form button[type="submit"]').click();

      // Wait for login to complete
      cy.contains("Hi, testuser", { timeout: 2000 }).should("be.visible");

      // Navigate to account tab
      cy.navigateToTab("account");
    });

    it("should display user profile", () => {
      cy.contains("testuser").should("be.visible");
    });

    it("should display streaming controls", () => {
      cy.contains(/Stream|Live/i).should("be.visible");
    });
  });

  describe("Streaming Controls", () => {
    it("should have start stream button", () => {
      cy.contains("button", /Start|Stream/i).should("exist");
    });

    it("should start streaming when button is clicked", () => {
      cy.mockTauriInvoke("stream_polymarket_prices", { success: true });

      cy.contains("button", /Start|Stream/i).click();

      // Should show streaming state
      cy.contains(/Stop|Streaming/i).should("be.visible");
    });

    it("should stop streaming when stop is clicked", () => {
      cy.mockTauriInvoke("stream_polymarket_prices", { success: true });
      cy.mockTauriInvoke("stop_polymarket_stream", { success: true });

      // Start
      cy.contains("button", /Start|Stream/i).click();

      // Stop
      cy.contains("button", /Stop/i).click();

      // Should show start button again
      cy.contains("button", /Start|Stream/i).should("be.visible");
    });
  });

  describe("Portfolio Section", () => {
    it("should display portfolio value", () => {
      cy.contains(/Portfolio|Balance|Value/i).should("be.visible");
    });

    it("should display positions if any", () => {
      // Positions section
      cy.contains(/Position|Holdings/i).should("be.visible");
    });
  });

  describe("Activity History", () => {
    it("should display recent activity", () => {
      cy.contains(/Activity|History|Recent/i).should("be.visible");
    });
  });

  describe("Settings", () => {
    it("should have settings section", () => {
      cy.contains(/Settings|Preferences/i).should("be.visible");
    });

    it("should allow toggling notifications", () => {
      // Notification toggle
      cy.get('input[type="checkbox"]').should("exist");
    });
  });

  describe("Logout", () => {
    beforeEach(() => {
      cy.mockTauriInvoke("login", {
        success: true,
        message: "Login successful",
        username: "testuser",
      });

      cy.contains("button", "Login").click();
      cy.get('input[placeholder="Enter username"]').type("testuser");
      cy.get('input[placeholder="Enter password"]').type("password123");
      cy.get('form button[type="submit"]').click();

      cy.contains("Hi, testuser", { timeout: 2000 }).should("be.visible");
    });

    it("should logout from account tab", () => {
      cy.contains("Logout").click();

      // Should be logged out
      cy.contains("button", "Login").should("be.visible");
    });
  });
});
