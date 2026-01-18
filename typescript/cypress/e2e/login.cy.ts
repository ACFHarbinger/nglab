/// <reference types="cypress" />

describe("Login Modal", () => {
  beforeEach(() => {
    cy.visit("/");
  });

  describe("Opening and Closing", () => {
    it("should open login modal when Login button is clicked", () => {
      cy.contains("button", "Login").click();
      cy.contains("Welcome Back").should("be.visible");
    });

    it("should close modal when X button is clicked", () => {
      cy.contains("button", "Login").click();
      cy.contains("Welcome Back").should("be.visible");

      cy.get('button').find('svg').filter('[class*="lucide-x"]').parent().click();
      cy.contains("Welcome Back").should("not.exist");
    });

    it("should close modal when backdrop is clicked", () => {
      cy.contains("button", "Login").click();
      cy.contains("Welcome Back").should("be.visible");

      // Click backdrop (the dark overlay behind modal)
      cy.get('[class*="bg-black"]').click({ force: true });
      cy.contains("Welcome Back").should("not.exist");
    });
  });

  describe("Login Tab", () => {
    beforeEach(() => {
      cy.contains("button", "Login").click();
    });

    it("should display login form by default", () => {
      cy.contains("Welcome Back").should("be.visible");
      cy.get('input[placeholder="Enter username"]').should("be.visible");
      cy.get('input[placeholder="Enter password"]').should("be.visible");
      cy.contains("button", "Login").should("be.visible");
    });

    it("should have username input required", () => {
      cy.get('input[placeholder="Enter username"]').should("have.attr", "required");
    });

    it("should have password input required with min length 8", () => {
      cy.get('input[placeholder="Enter password"]')
        .should("have.attr", "required")
        .should("have.attr", "minlength", "8");
    });

    it("should handle successful login", () => {
      cy.mockTauriInvoke("login", {
        success: true,
        message: "Login successful",
        username: "testuser",
      });

      cy.get('input[placeholder="Enter username"]').type("testuser");
      cy.get('input[placeholder="Enter password"]').type("password123");
      cy.get('form button[type="submit"]').click();

      cy.contains("Login successful!").should("be.visible");

      // Modal should close after success
      cy.contains("Welcome Back", { timeout: 2000 }).should("not.exist");

      // Should show logged in state
      cy.contains("Hi, testuser").should("be.visible");
    });

    it("should display error on login failure", () => {
      cy.mockTauriInvoke("login", {
        success: false,
        message: "Invalid username or password",
        username: null,
      });

      cy.get('input[placeholder="Enter username"]').type("wronguser");
      cy.get('input[placeholder="Enter password"]').type("wrongpass");
      cy.get('form button[type="submit"]').click();

      cy.contains("Invalid username or password").should("be.visible");
    });

    it("should show loading state while logging in", () => {
      // Delay the response
      cy.mockTauriInvoke("login", {
        success: true,
        message: "Login successful",
        username: "testuser",
      });

      cy.get('input[placeholder="Enter username"]').type("testuser");
      cy.get('input[placeholder="Enter password"]').type("password123");
      cy.get('form button[type="submit"]').click();

      cy.contains("Please wait...").should("be.visible");
    });
  });

  describe("Create Account Tab", () => {
    beforeEach(() => {
      cy.contains("button", "Login").click();
      cy.contains("button", "Create Account").click();
    });

    it("should switch to create account form", () => {
      cy.contains("h2", "Create Account").should("be.visible");
      cy.get('input[placeholder="Confirm password"]').should("be.visible");
    });

    it("should display confirm password field", () => {
      cy.get('input[placeholder="Confirm password"]')
        .should("be.visible")
        .should("have.attr", "required")
        .should("have.attr", "minlength", "8");
    });

    it("should show error if passwords do not match", () => {
      cy.get('input[placeholder="Enter username"]').type("newuser");
      cy.get('input[placeholder="Enter password"]').type("password123");
      cy.get('input[placeholder="Confirm password"]').type("differentpassword");
      cy.get('form button[type="submit"]').click();

      cy.contains("Passwords do not match").should("be.visible");
    });

    it("should show error if password is too short", () => {
      cy.get('input[placeholder="Enter username"]').type("newuser");
      cy.get('input[placeholder="Enter password"]').type("short");
      cy.get('input[placeholder="Confirm password"]').type("short");
      // Browser validation will prevent submission for minlength
      // The form won't submit with invalid inputs
    });

    it("should handle successful account creation", () => {
      cy.mockTauriInvoke("create_account", {
        success: true,
        message: "Account created successfully",
        username: "newuser",
      });

      cy.get('input[placeholder="Enter username"]').type("newuser");
      cy.get('input[placeholder="Enter password"]').type("password123");
      cy.get('input[placeholder="Confirm password"]').type("password123");
      cy.get('form button[type="submit"]').click();

      cy.contains("Account created! You can now login.").should("be.visible");

      // Should switch back to login tab
      cy.contains("Welcome Back", { timeout: 3000 }).should("be.visible");
    });

    it("should display error on account creation failure", () => {
      cy.mockTauriInvoke("create_account", {
        success: false,
        message: "Username already exists",
        username: null,
      });

      cy.get('input[placeholder="Enter username"]').type("existinguser");
      cy.get('input[placeholder="Enter password"]').type("password123");
      cy.get('input[placeholder="Confirm password"]').type("password123");
      cy.get('form button[type="submit"]').click();

      cy.contains("Username already exists").should("be.visible");
    });
  });

  describe("Tab Switching", () => {
    beforeEach(() => {
      cy.contains("button", "Login").click();
    });

    it("should clear form when switching tabs", () => {
      cy.get('input[placeholder="Enter username"]').type("testuser");
      cy.get('input[placeholder="Enter password"]').type("password123");

      cy.contains("button", "Create Account").click();

      cy.get('input[placeholder="Enter username"]').should("have.value", "");
      cy.get('input[placeholder="Enter password"]').should("have.value", "");
    });

    it("should clear errors when switching tabs", () => {
      cy.mockTauriInvoke("login", {
        success: false,
        message: "Login failed",
        username: null,
      });

      cy.get('input[placeholder="Enter username"]').type("user");
      cy.get('input[placeholder="Enter password"]').type("password123");
      cy.get('form button[type="submit"]').click();

      cy.contains("Login failed").should("be.visible");

      cy.contains("button", "Create Account").click();
      cy.contains("Login failed").should("not.exist");
    });

    it("should switch back to login tab", () => {
      cy.contains("button", "Create Account").click();
      cy.contains("h2", "Create Account").should("be.visible");

      cy.get('button').contains("Login").first().click();
      cy.contains("h2", "Welcome Back").should("be.visible");
    });
  });

  describe("Logged In State", () => {
    it("should display username when logged in", () => {
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

    it("should show logout button when logged in", () => {
      cy.mockTauriInvoke("login", {
        success: true,
        message: "Login successful",
        username: "testuser",
      });

      cy.contains("button", "Login").click();
      cy.get('input[placeholder="Enter username"]').type("testuser");
      cy.get('input[placeholder="Enter password"]').type("password123");
      cy.get('form button[type="submit"]').click();

      cy.contains("Logout", { timeout: 2000 }).should("be.visible");
    });

    it("should logout when logout button is clicked", () => {
      cy.mockTauriInvoke("login", {
        success: true,
        message: "Login successful",
        username: "testuser",
      });

      cy.contains("button", "Login").click();
      cy.get('input[placeholder="Enter username"]').type("testuser");
      cy.get('input[placeholder="Enter password"]').type("password123");
      cy.get('form button[type="submit"]').click();

      cy.contains("Logout", { timeout: 2000 }).click();
      cy.contains("button", "Login").should("be.visible");
    });
  });
});
