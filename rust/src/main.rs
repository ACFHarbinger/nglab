/*!
 * nglab-tui - Real-time terminal dashboard for the trading arena.
 *
 * This binary provides a TUI built with `ratatui` to visualize
 * the order book, portfolio stats, and price action from the Rust backend.
 */

use color_eyre::Result;
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use nglab::simulation::gym::{ActionType, StepInfo, TradingEnv};
use ratatui::{prelude::*, widgets::*};
use std::io;
use std::time::{Duration, Instant};

/**
 * App state for the TUI dashboard.
 *
 * Contains the environment instance, current step information,
 * and user interface state.
 */
struct App {
    /** The underlying trading environment */
    env: TradingEnv,
    /** Latest step information and metrics */
    info: StepInfo,
    /** Historical price data for visualization */
    price_history: Vec<f64>,
    /** Flag to signal application termination */
    should_quit: bool,
    /** String description of the last user action */
    last_action: String,
}

impl App {
    /**
     * Create a new instance of the TUI application.
     */
    fn new() -> Self {
        let mut env = TradingEnv::new(10000.0, 0.001, 30, 2000, true, None);

        // Generate dummy price data for demonstration
        let mut prices = Vec::new();
        let mut p = 100.0;
        for _ in 0..2100 {
            p += (rand::random::<f64>() - 0.5) * 2.0;
            prices.push(p);
        }
        env.load_prices(prices);

        env.reset_rs();

        App {
            env,
            info: StepInfo::default(),
            price_history: Vec::new(),
            should_quit: false,
            last_action: "None".to_string(),
        }
    }

    /**
     * Advance the simulation by one step with the given action.
     */
    fn step(&mut self, action: i32) {
        let (obs, _reward, _term, _trunc, info) = self.env.step_rs(action);
        self.info = info;
        // The first feature in our observation is normalized price
        // Let's get the absolute price from the environment logic if available,
        // or just use the last value from dummy prices for viz.
        if let Some(price) = obs.get(obs.len() - 6) {
            // This is normalized, but for sparkline it's fine
            self.price_history.push(*price);
        }
        if self.price_history.len() > 100 {
            self.price_history.remove(0);
        }

        self.last_action = match ActionType::from(action) {
            ActionType::Hold => "Hold",
            ActionType::Buy => "Buy",
            ActionType::Sell => "Sell",
        }
        .to_string();
    }
}

/**
 * Main entry point for the TUI application.
 */
fn main() -> Result<()> {
    // Setup terminal
    color_eyre::install()?;
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create app
    let mut app = App::new();

    // Initial step to populate data
    app.step(0);

    let tick_rate = Duration::from_millis(100);
    let mut last_tick = Instant::now();

    loop {
        terminal.draw(|f| ui(f, &app))?;

        let timeout = tick_rate.saturating_sub(last_tick.elapsed());
        if event::poll(timeout)? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => app.should_quit = true,
                        KeyCode::Char(' ') => app.step(0), // Hold/Step
                        KeyCode::Char('b') => app.step(1), // Buy
                        KeyCode::Char('s') => app.step(2), // Sell
                        _ => {}
                    }
                }
            }
        }

        if last_tick.elapsed() >= tick_rate {
            last_tick = Instant::now();
        }

        if app.should_quit {
            break;
        }
    }

    // Restore terminal
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    Ok(())
}

/**
 * Render the TUI components.
 */
fn ui(f: &mut Frame, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Header
            Constraint::Min(10),   // Main body
            Constraint::Length(3), // Footer
        ])
        .split(f.area());

    // Header
    let header_widget = Paragraph::new(Line::from(vec![
        Span::styled(
            " nglab ",
            Style::default()
                .add_modifier(Modifier::BOLD)
                .fg(Color::Yellow),
        ),
        Span::raw("- High Frequency Trading Arena "),
        Span::styled(
            "v0.1.0",
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::ITALIC),
        ),
    ]))
    .block(Block::bordered().border_type(BorderType::Rounded))
    .alignment(Alignment::Left);

    f.render_widget(header_widget, chunks[0]);

    // Main Body
    let body_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(30), // Stats
            Constraint::Percentage(70), // Charts / Book
        ])
        .split(chunks[1]);

    // Stats
    let stats_text = vec![
        Line::from(vec![
            Span::raw("Portfolio Value: "),
            Span::styled(
                format!("{:.2}", app.info.portfolio_value),
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::raw("Cash:            "),
            Span::styled(
                format!("{:.2}", app.info.cash),
                Style::default().fg(Color::Green),
            ),
        ]),
        Line::from(vec![
            Span::raw("Position:        "),
            Span::styled(
                format!("{:.4}", app.info.position),
                Style::default().fg(Color::Magenta),
            ),
        ]),
        Line::from(vec![
            Span::raw("Sharpe Ratio:    "),
            Span::styled(
                format!("{:.4}", app.info.sharpe_ratio),
                Style::default().fg(Color::Blue),
            ),
        ]),
        Line::from(vec![
            Span::raw("Total Steps:     "),
            Span::styled(
                format!("{}", app.info.total_steps),
                Style::default().fg(Color::Gray),
            ),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("Last Action:     "),
            Span::styled(
                &app.last_action,
                Style::default().fg(if app.last_action == "Buy" {
                    Color::Green
                } else if app.last_action == "Sell" {
                    Color::Red
                } else {
                    Color::Gray
                }),
            ),
        ]),
    ];

    let stats = Paragraph::new(stats_text)
        .block(
            Block::default()
                .title(" Portfolio Stats ")
                .borders(Borders::ALL),
        )
        .alignment(Alignment::Left);
    f.render_widget(stats, body_chunks[0]);

    // Right Side (Visuals)
    let right_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(60), // Order Book
            Constraint::Percentage(40), // Price Chart
        ])
        .split(body_chunks[1]);

    // Order Book (Simplified view for TUI)
    let book = app.env.orderbook();
    let bids = book.bid_depth(5);
    let asks = book.ask_depth(5);

    let mut book_rows = Vec::new();
    // Asks (Red) - reverse order to show best ask closest to mid
    for (p, q) in asks.iter().rev() {
        book_rows.push(Row::new(vec![
            Cell::from("").style(Style::default()),
            Cell::from(format!("{:.2}", p)).style(Style::default().fg(Color::Red)),
            Cell::from(format!("{:.2}", q)).style(Style::default().fg(Color::DarkGray)),
        ]));
    }
    // Mid Price
    if let Some(mid) = book.mid_price() {
        book_rows.push(Row::new(vec![
            Cell::from(" Mid ->")
                .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::DIM)),
            Cell::from(format!("{:.2}", mid)).style(
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Cell::from("").style(Style::default()),
        ]));
    }
    // Bids (Green)
    for (p, q) in bids {
        book_rows.push(Row::new(vec![
            Cell::from(format!("{:.2}", q)).style(Style::default().fg(Color::DarkGray)),
            Cell::from(format!("{:.2}", p)).style(Style::default().fg(Color::Green)),
            Cell::from("").style(Style::default()),
        ]));
    }

    let table = Table::new(
        book_rows,
        [
            Constraint::Percentage(33),
            Constraint::Percentage(33),
            Constraint::Percentage(33),
        ],
    )
    .block(Block::default().title(" Order Book ").borders(Borders::ALL))
    .header(
        Row::new(vec!["Bids Vol", "Price", "Asks Vol"])
            .style(Style::default().add_modifier(Modifier::BOLD)),
    );

    f.render_widget(table, right_chunks[0]);

    // Price Chart (Sparkline)
    let sparkline = Sparkline::default()
        .block(
            Block::default()
                .title(" Price Action (Lookback) ")
                .borders(Borders::ALL),
        )
        .data(
            app.price_history
                .iter()
                .map(|v| (v * 10.0) as u64)
                .collect::<Vec<_>>(),
        ) // Scale for visibility
        .style(Style::default().fg(Color::Green));
    f.render_widget(sparkline, right_chunks[1]);

    // Footer
    let footer_text = " [SPACE] Step | [B] Buy | [S] Sell | [Q/ESC] Quit ";
    let footer = Paragraph::new(footer_text)
        .block(Block::default().borders(Borders::ALL))
        .alignment(Alignment::Center)
        .style(Style::default().fg(Color::DarkGray));
    f.render_widget(footer, chunks[2]);
}
