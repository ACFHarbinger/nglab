/// <reference types="cypress" />

describe("Vault Tab", () => {
  beforeEach(() => {
    cy.visit("/");
    cy.navigateToTab("vault");
  });

  describe("Layout", () => {
    it("should display vault tab content", () => {
      cy.contains("button", "Vault").should("have.class", "text-white");
    });

    it("should display vault title", () => {
      cy.contains(/Vault|Secrets|Security/i).should("be.visible");
    });
  });

  describe("Vault Locked State", () => {
    beforeEach(() => {
      cy.mockTauriInvoke("is_vault_unlocked", {
        success: true,
        data: false,
      });
    });

    it("should display unlock interface when vault is locked", () => {
      cy.contains(/Unlock|Password|Master/i).should("be.visible");
    });

    it("should have password input for unlocking", () => {
      cy.get('input[type="password"]').should("exist");
    });

    it("should have unlock button", () => {
      cy.contains("button", /Unlock/i).should("be.visible");
    });

    it("should unlock vault with correct password", () => {
      cy.mockTauriInvoke("unlock_vault", {
        success: true,
        message: "Vault unlocked",
      });

      cy.get('input[type="password"]').type("correctpassword");
      cy.contains("button", /Unlock/i).click();

      cy.contains("Vault unlocked", { timeout: 5000 }).should("be.visible");
    });

    it("should show error on incorrect password", () => {
      cy.mockTauriInvoke("unlock_vault", {
        success: false,
        message: "Invalid password",
      });

      cy.get('input[type="password"]').type("wrongpassword");
      cy.contains("button", /Unlock/i).click();

      cy.contains("Invalid password", { timeout: 5000 }).should("be.visible");
    });
  });

  describe("Vault Unlocked State", () => {
    beforeEach(() => {
      cy.mockTauriInvoke("is_vault_unlocked", {
        success: true,
        data: true,
      });

      cy.mockTauriInvoke("list_vault_secrets", {
        success: true,
        data: [
          { id: 1, name: "API Key 1", service: "polymarket" },
          { id: 2, name: "API Key 2", service: "exchange" },
        ],
      });
    });

    it("should display secrets list when unlocked", () => {
      cy.contains(/Secrets|Credentials|Keys/i).should("be.visible");
    });

    it("should display existing secrets", () => {
      cy.contains("API Key 1").should("be.visible");
    });

    it("should have add secret button", () => {
      cy.contains("button", /Add|New|Create/i).should("be.visible");
    });

    it("should have lock button", () => {
      cy.contains("button", /Lock/i).should("be.visible");
    });
  });

  describe("Adding Secrets", () => {
    beforeEach(() => {
      cy.mockTauriInvoke("is_vault_unlocked", {
        success: true,
        data: true,
      });

      cy.mockTauriInvoke("list_vault_secrets", {
        success: true,
        data: [],
      });
    });

    it("should open add secret form", () => {
      cy.contains("button", /Add|New|Create/i).click();
      cy.get('input[placeholder*="name"], input[placeholder*="Name"]').should("exist");
    });

    it("should add a new secret", () => {
      cy.mockTauriInvoke("add_vault_secret", {
        success: true,
        message: "Secret added",
        data: 1,
      });

      cy.contains("button", /Add|New|Create/i).click();

      cy.get('input').first().type("My API Key");
      cy.get('input').eq(1).type("secret_value_123");

      cy.contains("button", /Save|Add|Submit/i).click();

      cy.contains("Secret added", { timeout: 5000 }).should("be.visible");
    });
  });

  describe("Managing Secrets", () => {
    beforeEach(() => {
      cy.mockTauriInvoke("is_vault_unlocked", {
        success: true,
        data: true,
      });

      cy.mockTauriInvoke("list_vault_secrets", {
        success: true,
        data: [
          { id: 1, name: "Test Secret", service: "test" },
        ],
      });
    });

    it("should view secret details", () => {
      cy.mockTauriInvoke("get_vault_secret", {
        success: true,
        data: {
          id: 1,
          name: "Test Secret",
          value: "secret_value",
          service: "test",
        },
      });

      cy.contains("Test Secret").click();
      // Should show secret details
    });

    it("should delete a secret", () => {
      cy.mockTauriInvoke("delete_vault_secret", {
        success: true,
        message: "Secret deleted",
      });

      cy.get('button[title*="Delete"], button[aria-label*="delete"]').first().click();

      // Confirm deletion
      cy.contains("button", /Confirm|Yes|Delete/i).click();

      cy.contains("Secret deleted", { timeout: 5000 }).should("be.visible");
    });
  });

  describe("Locking Vault", () => {
    beforeEach(() => {
      cy.mockTauriInvoke("is_vault_unlocked", {
        success: true,
        data: true,
      });

      cy.mockTauriInvoke("list_vault_secrets", {
        success: true,
        data: [],
      });
    });

    it("should lock vault when lock button is clicked", () => {
      cy.mockTauriInvoke("lock_vault", {
        success: true,
        message: "Vault locked",
      });

      cy.contains("button", /Lock/i).click();

      // Should show locked state
      cy.contains(/Unlock|Password/i, { timeout: 5000 }).should("be.visible");
    });
  });

  describe("Integrations", () => {
    beforeEach(() => {
      cy.mockTauriInvoke("is_vault_unlocked", {
        success: true,
        data: true,
      });

      cy.mockTauriInvoke("list_integrations", {
        success: true,
        data: [
          { id: 1, service_name: "polymarket", created_at: "2026-01-15" },
        ],
      });
    });

    it("should display integrations section", () => {
      cy.contains(/Integration|Connect/i).should("be.visible");
    });

    it("should list connected integrations", () => {
      cy.contains("polymarket").should("be.visible");
    });

    it("should add Polymarket integration", () => {
      cy.mockTauriInvoke("save_polymarket_integration", {
        success: true,
        message: "Integration saved",
        data: 2,
      });

      cy.contains("button", /Connect|Add/i).click();

      // Fill integration form
      cy.get('input[placeholder*="API"]').type("api_key_123");
      cy.get('input[placeholder*="Secret"]').type("secret_123");
      cy.get('input[placeholder*="Passphrase"]').type("passphrase_123");

      cy.contains("button", /Save|Connect/i).click();

      cy.contains("Integration saved", { timeout: 5000 }).should("be.visible");
    });

    it("should delete integration", () => {
      cy.mockTauriInvoke("delete_integration", {
        success: true,
        message: "Integration deleted",
      });

      cy.get('button[title*="Delete"], button[aria-label*="delete"]').first().click();
      cy.contains("button", /Confirm|Yes/i).click();

      cy.contains("Integration deleted", { timeout: 5000 }).should("be.visible");
    });
  });
});
