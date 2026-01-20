describe('Visual Regression', () => {
    beforeEach(() => {
        // Visit the home page
        cy.visit('/');

        // Wait for the app to load - check for a key element
        // Adjust selector based on actual app content
        cy.get('main', { timeout: 10000 }).should('be.visible');
    });

    it('Dashboard Overview', () => {
        // Allow animations to settle
        cy.wait(1000);
        cy.screenshot('dashboard-overview');
    });

    it('Trading Terminal', () => {
        // Navigate if necessary, or just check the main view if it's there
        // Assuming Trading Terminal is part of the main dashboard or a tab
        // cy.contains('Terminal').click();
        cy.screenshot('trading-terminal');
    });

    it('Responsive Mobile', () => {
        cy.viewport('iphone-x');
        cy.wait(500);
        cy.screenshot('dashboard-mobile');
    });
});
