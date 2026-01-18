/// <reference types="cypress" />

import "./commands";
import { mount } from "cypress/react";

// Augment the Cypress namespace to include mount
declare global {
  namespace Cypress {
    interface Chainable {
      mount: typeof mount;
    }
  }
}

Cypress.Commands.add("mount", mount);

// Mock Tauri API for component tests
beforeEach(() => {
  // Create mock invoke function
  const mockInvoke = cy.stub().as("tauriInvoke");

  // Set default responses
  mockInvoke.resolves({ success: true, data: null });

  // Inject into window before component mounts
  cy.window({ log: false }).then((win) => {
    (win as any).__TAURI__ = {
      core: {
        invoke: mockInvoke,
      },
    };
    (win as any).__TAURI_INTERNALS__ = {
      invoke: mockInvoke,
    };
  });
});
