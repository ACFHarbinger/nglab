/// <reference types="cypress" />

describe("Navigation", () => {
  beforeEach(() => {
    cy.visit("/");
  });

  describe("Header Navigation", () => {
    it("should display the app logo and title", () => {
      cy.get("header").within(() => {
        cy.contains("NGL").should("be.visible");
        cy.contains(".trade").should("be.visible");
      });
    });

    it("should have Dashboard selected by default", () => {
      cy.contains("button", "Dashboard").should(
        "have.class",
        "text-white"
      );
    });

    it("should navigate to Intelligence tab", () => {
      cy.contains("button", "Intelligence").click();
      cy.contains("button", "Intelligence").should(
        "have.class",
        "text-white"
      );
    });

    it("should navigate to Markets tab", () => {
      cy.contains("button", "Markets").click();
      cy.contains("button", "Markets").should(
        "have.class",
        "text-white"
      );
    });

    it("should navigate to Positions tab", () => {
      cy.contains("button", "Positions").click();
      cy.contains("button", "Positions").should(
        "have.class",
        "text-white"
      );
    });

    it("should navigate to Favorites tab", () => {
      cy.contains("button", "Favorites").click();
      cy.contains("button", "Favorites").should(
        "have.class",
        "text-white"
      );
      // Should see favorites content
      cy.contains("Favorite Markets").should("be.visible");
    });

    it("should navigate to Account tab", () => {
      cy.contains("button", "Account").click();
      cy.contains("button", "Account").should(
        "have.class",
        "text-white"
      );
    });

    it("should navigate to Simulation tab", () => {
      cy.contains("button", "Simulation").click();
      cy.contains("button", "Simulation").should(
        "have.class",
        "text-white"
      );
      // Simulation controls should appear
      cy.contains("button", "Start").should("be.visible");
    });

    it("should navigate to Scraper tab", () => {
      cy.contains("button", "Scraper").click();
      cy.contains("button", "Scraper").should(
        "have.class",
        "text-white"
      );
    });

    it("should navigate to Pricing tab", () => {
      cy.contains("button", "Pricing").click();
      cy.contains("button", "Pricing").should(
        "have.class",
        "text-white"
      );
    });

    it("should navigate to Training tab", () => {
      cy.contains("button", "Training").click();
      cy.contains("button", "Training").should(
        "have.class",
        "text-white"
      );
    });

    it("should navigate to News tab", () => {
      cy.contains("button", "News").click();
      cy.contains("button", "News").should(
        "have.class",
        "text-white"
      );
    });

    it("should navigate to Vault tab", () => {
      cy.contains("button", "Vault").click();
      cy.contains("button", "Vault").should(
        "have.class",
        "text-white"
      );
    });
  });

  describe("Search Bar", () => {
    it("should display search input in header", () => {
      cy.get('input[placeholder*="search"]').should("be.visible");
    });
  });

  describe("Login Button", () => {
    it("should display login button when not logged in", () => {
      cy.contains("button", "Login").should("be.visible");
    });

    it("should open login modal when clicked", () => {
      cy.contains("button", "Login").click();
      // Modal should appear - checking for common modal elements
      cy.get('[role="dialog"], [class*="modal"]').should("exist");
    });
  });
});
