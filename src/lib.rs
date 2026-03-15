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
    use std::collections::HashMap;
    use std::thread::JoinHandle;

    #[derive(Debug)]
    struct Layer {
        name: String,
        grad_std: f64,
        data_std: f64,
    }

    impl Layer {
        fn log_ratio(&self) -> f64 {
            if self.data_std == 0.0 {
                return f64::NEG_INFINITY;
            }
            (self.grad_std / self.data_std).log10()
        }
    }

    struct Stats {
        loss: f64,
        layers: Vec<Layer>,
    }

    enum Message {
        Stats(Stats),
        Quit,
    }

    struct App {
        rx: Receiver<Message>,
        loss_data: Vec<(f64, f64)>,
        ratio_history: HashMap<String, Vec<(f64, f64)>>,
        step: usize,
    }

    impl App {
        fn new(rx: Receiver<Message>) -> Self {
            App {
                rx,
                loss_data: Vec::new(),
                ratio_history: HashMap::new(),
                step: 0,
            }
        }

        fn run(&mut self, terminal: &mut DefaultTerminal) -> std::io::Result<()> {
            // drain any buffered input events before starting
            while crossterm::event::poll(std::time::Duration::from_millis(0))? {
                crossterm::event::read()?;
            }

            loop {
                while let Ok(msg) = self.rx.try_recv() {
                    match msg {
                        Message::Stats(stats) => {
                            self.loss_data.push((self.step as f64, stats.loss));
                            self.step += 1;
                            if self.loss_data.len() > 100 {
                                self.loss_data.remove(0);
                            }

                            for layer in &stats.layers {
                                let history = self
                                    .ratio_history
                                    .entry(layer.name.clone())
                                    .or_insert_with(Vec::new);
                                history.push((self.step as f64, layer.log_ratio()));
                                if history.len() > 100 {
                                    history.remove(0);
                                }
                            }
                        }
                        Message::Quit => return Ok(()),
                    }
                }

                terminal.draw(|frame| {
                    let [top, bottom] = ratatui::layout::Layout::vertical(
                        [ratatui::layout::Constraint::Fill(1); 2],
                    )
                    .areas(frame.area());
                    render_loss_chart(frame, top, &self.loss_data);
                    render_ratio_chart(frame, bottom, &self.ratio_history, self.step);
                })?;

                if crossterm::event::poll(std::time::Duration::from_millis(16))? {
                    if crossterm::event::read()?.is_key_press() {
                        break Ok(());
                    }
                }
            }
        }
    }

    fn render_ratio_chart(
        frame: &mut ratatui::Frame,
        area: ratatui::layout::Rect,
        ratio_history: &HashMap<String, Vec<(f64, f64)>>,
        step: usize,
    ) {
        let x_min = step.saturating_sub(100) as f64;
        let x_max = (step as f64).max(x_min + 1.0); // avoid x_min == x_max
        let reference_line: Vec<(f64, f64)> = vec![(x_min, -3.0), (x_max, -3.0)];

        let colors = [
            Color::Cyan,
            Color::Yellow,
            Color::Green,
            Color::Magenta,
            Color::Red,
            Color::Blue,
            Color::LightCyan,
            Color::LightYellow,
        ];

        // collect into a sorted vec so layer order is stable across frames
        let mut entries: Vec<(&String, &Vec<(f64, f64)>)> = ratio_history.iter().collect();
        entries.sort_by_key(|(name, _)| *name);

        let mut datasets: Vec<Dataset> = entries
            .iter()
            .enumerate()
            .map(|(i, (name, data))| {
                Dataset::default()
                    .name(name.as_str())
                    .marker(symbols::Marker::Braille)
                    .graph_type(GraphType::Line)
                    .style(Style::default().fg(colors[i % colors.len()]))
                    .data(data)
            })
            .collect();

        datasets.push(
            Dataset::default()
                .name("-3 (target)")
                .marker(symbols::Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(Color::White))
                .data(&reference_line),
        );

        let chart = Chart::new(datasets)
            .block(Block::bordered().title("Update/Data Ratio (log10) — target: -3"))
            .x_axis(
                Axis::default()
                    .title("Steps")
                    .style(Style::default().fg(Color::Gray))
                    .labels([
                        format!("{:.0}", x_min),
                        format!("{:.0}", (x_min + x_max) / 2.0),
                        format!("{:.0}", x_max),
                    ])
                    .bounds([x_min, x_max]),
            )
            .y_axis(
                Axis::default()
                    .title("log10(ratio)")
                    .style(Style::default().fg(Color::Gray))
                    .labels(["-4", "-3", "-2", "-1", "0"])
                    .bounds([-4.0, 0.0]),
            );

        frame.render_widget(chart, area);
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
                .name("loss")
                .marker(symbols::Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(Color::Cyan))
                .data(data),
        ];

        let chart = Chart::new(datasets)
            .block(Block::bordered().title("Loss"))
            .x_axis(
                Axis::default()
                    .title("Steps")
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
                    let _ = std::fs::write("/tmp/gradtop_error.txt", format!("{e}"));
                }
            });

            Monitor {
                sender: tx,
                handle: Some(handle),
            }
        }

        fn tick(
            &self,
            loss: f64,
            names: Vec<String>,
            grad_stds: Vec<f64>,
            data_stds: Vec<f64>,
        ) -> PyResult<()> {
            let layers: Vec<Layer> = names
                .into_iter()
                .zip(grad_stds)
                .zip(data_stds)
                .map(|((name, grad_std), data_std)| Layer {
                    name,
                    grad_std,
                    data_std,
                })
                .collect();
            let stats = Stats { loss, layers };
            let _ = self.sender.send(Message::Stats(stats));
            Ok(())
        }
    }

    impl Drop for Monitor {
        fn drop(&mut self) {
            let _ = self.sender.send(Message::Quit);
            if let Some(handle) = self.handle.take() {
                let _ = handle.join();
            }
        }
    }
}
