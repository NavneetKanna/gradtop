use pyo3::prelude::*;
use std::sync::mpsc::{self, Receiver, Sender};

#[pymodule]
mod gradtop {
    use super::*;
    use ratatui::{
        DefaultTerminal,
        style::{Color, Style},
        symbols,
        widgets::{Axis, Block, Chart, Dataset, GraphType},
    };
    use std::thread::JoinHandle;

    enum Message {
        Loss(f64),
        Quit,
    }

    struct App {
        rx: Receiver<Message>,
        loss_data: Vec<(f64, f64)>,
        step: usize,
    }

    impl App {
        fn new(rx: Receiver<Message>) -> Self {
            App {
                rx,
                loss_data: Vec::new(),
                step: 0,
            }
        }

        fn run(&mut self, terminal: &mut DefaultTerminal) -> std::io::Result<()> {
            loop {
                while let Ok(msg) = self.rx.try_recv() {
                    match msg {
                        Message::Loss(loss) => {
                            self.loss_data.push((self.step as f64, loss));
                            self.step += 1;
                            if self.loss_data.len() > 30 {
                                self.loss_data.remove(0);
                            }
                        }
                        Message::Quit => return Ok(()),
                    }
                }
                terminal.draw(|frame| {
                    let [top, _bottom] = ratatui::layout::Layout::vertical(
                        [ratatui::layout::Constraint::Fill(1); 2],
                    )
                    .areas(frame.area());
                    render_loss_chart(frame, top, &self.loss_data);
                })?;

                if crossterm::event::poll(std::time::Duration::from_millis(16))? {
                    if crossterm::event::read()?.is_key_press() {
                        break Ok(());
                    }
                }
            }
        }
    }

    fn render_loss_chart(
        frame: &mut ratatui::Frame,
        area: ratatui::layout::Rect,
        data: &[(f64, f64)],
    ) {
        let x_bounds = [
            data.first().map(|p| p.0).unwrap_or(0.0),
            data.last().map(|p| p.0).unwrap_or(1.0),
        ];
        let y_bounds = [
            data.iter().map(|p| p.1).fold(f64::INFINITY, f64::min),
            data.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max),
        ];

        let datasets = vec![
            Dataset::default()
                .name("testloss")
                .marker(symbols::Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(Color::Cyan))
                .data(data),
        ];

        let chart = Chart::new(datasets)
            .block(Block::bordered())
            .x_axis(
                Axis::default()
                    .title("Epoch")
                    .style(Style::default().fg(Color::Gray))
                    .labels([
                        format!("{:.0}", x_bounds[0]),
                        format!("{:.0}", (x_bounds[0] + x_bounds[1]) / 2.0),
                        format!("{:.0}", x_bounds[1]),
                    ])
                    .bounds(x_bounds),
            )
            .y_axis(
                Axis::default()
                    .title("nottestLoss")
                    .style(Style::default().fg(Color::Gray))
                    .labels([
                        format!("{:.3}", y_bounds[0]),
                        format!("{:.3}", (y_bounds[0] + y_bounds[1]) / 2.0),
                        format!("{:.3}", y_bounds[1]),
                    ])
                    .bounds(y_bounds),
            );

        frame.render_widget(chart, area);
    }

    #[pyclass]
    pub struct Monitor {
        sender: Sender<Message>,
        handle: Option<JoinHandle<()>>,
    }

    #[pymethods]
    impl Monitor {
        #[new]
        fn new() -> Self {
            let (tx, rx) = mpsc::channel::<Message>();

            let handle = std::thread::spawn(move || {
                let mut app = App::new(rx);
                if let Err(e) = ratatui::run(|terminal| app.run(terminal)) {
                    eprintln!("TUI thread error: {e}");
                }
            });

            Monitor {
                sender: tx,
                handle: Some(handle),
            }
        }

        fn tick(&self, loss: f64) -> PyResult<()> {
            let _ = self.sender.send(Message::Loss(loss));
            Ok(())
        }
    }

    impl Drop for Monitor {
        fn drop(&mut self) {
            // Signal the TUI thread to quit cleanly
            let _ = self.sender.send(Message::Quit);
            // Wait for it to finish restoring the terminal
            if let Some(handle) = self.handle.take() {
                let _ = handle.join();
            }
        }
    }
}
