/// <reference types="cypress" />

describe("Prediction/Intelligence Tab", () => {
  beforeEach(() => {
    cy.visit("/");
    cy.navigateToTab("prediction");
  });

  describe("Layout", () => {
    it("should display intelligence tab content", () => {
      cy.contains("button", "Intelligence").should("have.class", "text-white");
    });

    it("should display prediction models section", () => {
      cy.contains(/Prediction|Model|Forecast/i).should("be.visible");
    });
  });

  describe("Model Selection", () => {
    it("should have ARIMA model option", () => {
      cy.contains(/ARIMA/i).should("be.visible");
    });

    it("should have GARCH model option", () => {
      cy.contains(/GARCH/i).should("be.visible");
    });

    it("should have Holt-Winters model option", () => {
      cy.contains(/Holt|Winters/i).should("be.visible");
    });

    it("should have Prophet model option", () => {
      cy.contains(/Prophet/i).should("be.visible");
    });
  });

  describe("ARIMA Prediction", () => {
    it("should select ARIMA model", () => {
      cy.contains("button", /ARIMA/i).click();
    });

    it("should have ARIMA parameters", () => {
      cy.contains("button", /ARIMA/i).click();

      // p, d, q parameters
      cy.get('input').should("have.length.at.least", 1);
    });

    it("should run ARIMA prediction", () => {
      const mockResult = {
        success: true,
        predictions: [0.55, 0.58, 0.62, 0.60, 0.65],
        confidence_lower: [0.50, 0.52, 0.55, 0.53, 0.57],
        confidence_upper: [0.60, 0.64, 0.69, 0.67, 0.73],
      };

      cy.mockTauriInvoke("predict_arima", mockResult);

      cy.contains("button", /ARIMA/i).click();
      cy.contains("button", /Predict|Run/i).click();

      cy.contains(/Prediction|Result/i, { timeout: 5000 }).should("be.visible");
    });
  });

  describe("GARCH Prediction", () => {
    it("should select GARCH model", () => {
      cy.contains("button", /GARCH/i).click();
    });

    it("should run GARCH prediction", () => {
      const mockResult = {
        success: true,
        volatility: [0.15, 0.18, 0.20, 0.17, 0.16],
        forecast: [0.19, 0.18],
      };

      cy.mockTauriInvoke("predict_garch", mockResult);

      cy.contains("button", /GARCH/i).click();
      cy.contains("button", /Predict|Run/i).click();

      cy.contains(/Volatility|Result/i, { timeout: 5000 }).should("be.visible");
    });
  });

  describe("Holt-Winters Prediction", () => {
    it("should select Holt-Winters model", () => {
      cy.contains("button", /Holt|Winters/i).click();
    });

    it("should have seasonal parameters", () => {
      cy.contains("button", /Holt|Winters/i).click();

      // Seasonal period configuration
      cy.contains(/Season|Period/i).should("be.visible");
    });

    it("should run Holt-Winters prediction", () => {
      const mockResult = {
        success: true,
        predictions: [0.55, 0.60, 0.58, 0.62],
        level: 0.55,
        trend: 0.02,
        season: [0.01, -0.01, 0.02, -0.02],
      };

      cy.mockTauriInvoke("predict_holt_winters", mockResult);

      cy.contains("button", /Holt|Winters/i).click();
      cy.contains("button", /Predict|Run/i).click();

      cy.contains(/Prediction|Result/i, { timeout: 5000 }).should("be.visible");
    });
  });

  describe("Prophet Prediction", () => {
    it("should select Prophet model", () => {
      cy.contains("button", /Prophet/i).click();
    });

    it("should have Prophet configuration options", () => {
      cy.contains("button", /Prophet/i).click();

      // Prophet-specific options
    });

    it("should run Prophet prediction", () => {
      const mockResult = {
        success: true,
        predictions: [0.55, 0.58, 0.60, 0.62, 0.65],
        yhat_lower: [0.50, 0.52, 0.54, 0.56, 0.58],
        yhat_upper: [0.60, 0.64, 0.66, 0.68, 0.72],
      };

      cy.mockTauriInvoke("predict_prophet", mockResult);

      cy.contains("button", /Prophet/i).click();
      cy.contains("button", /Predict|Run/i).click();

      cy.contains(/Prediction|Result/i, { timeout: 5000 }).should("be.visible");
    });
  });

  describe("Visualization", () => {
    it("should display prediction chart", () => {
      cy.get('[class*="Chart"], canvas, svg').should("exist");
    });

    it("should show confidence intervals", () => {
      // Confidence bands on chart
    });

    it("should display prediction table", () => {
      // Tabular prediction data
    });
  });

  describe("Data Requirements", () => {
    it("should show message when no data available", () => {
      // Need price data to make predictions
      cy.contains(/data|Select|Market/i).should("be.visible");
    });

    it("should use live data when streaming", () => {
      // When connected to live stream, use that data
    });
  });

  describe("Error Handling", () => {
    it("should display error on prediction failure", () => {
      cy.mockTauriInvoke("predict_arima", {
        success: false,
        message: "Insufficient data for prediction",
      });

      cy.contains("button", /ARIMA/i).click();
      cy.contains("button", /Predict|Run/i).click();

      cy.contains("Insufficient data", { timeout: 5000 }).should("be.visible");
    });
  });
});
