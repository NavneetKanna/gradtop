use pyo3::prelude::*;
use std::sync::mpsc::{self, Sender};

#[pymodule]
mod gradtop {
    use super::*;
    use ratatui::{
        style::{Color, Style},
        symbols,
        widgets::{Axis, Block, Chart, Dataset, GraphType},
    };
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };

    #[pyclass]
    pub struct Monitor {
        sender: Sender<f64>,
        running: Arc<AtomicBool>,
        // store the thread handle so we can wait for it to finish
        thread: Option<std::thread::JoinHandle<()>>,
    }

    #[pymethods]
    impl Monitor {
        #[new]
        fn new() -> Self {
            let (tx, rx) = mpsc::channel::<f64>();
            let running = Arc::new(AtomicBool::new(true));
            let running_clone = running.clone();

            let handle = std::thread::spawn(move || {
                let mut terminal = ratatui::init();
                let mut loss_data: Vec<(f64, f64)> = Vec::new();
                let mut step = 0usize;

                loop {
                    while let Ok(loss) = rx.try_recv() {
                        loss_data.push((step as f64, loss));
                        step += 1;
                        if loss_data.len() > 30 {
                            loss_data.remove(0);
                        }
                    }

                    terminal
                        .draw(|frame| {
                            let [top, _bottom] = ratatui::layout::Layout::vertical(
                                [ratatui::layout::Constraint::Fill(1); 2],
                            )
                            .areas(frame.area());
                            render_loss_chart(frame, top, &loss_data);
                        })
                        .unwrap();

                    // check if Python side dropped the Monitor
                    if !running_clone.load(Ordering::Relaxed) {
                        break;
                    }

                    if crossterm::event::poll(std::time::Duration::from_millis(16)).unwrap() {
                        if crossterm::event::read().unwrap().is_key_press() {
                            break;
                        }
                    }
                }

                ratatui::restore();
            });

            Monitor {
                sender: tx,
                running,
                thread: Some(handle),
            }
        }

        fn tick(&self, loss: f64) -> PyResult<()> {
            let _ = self.sender.send(loss);
            Ok(())
        }

        fn is_running(&self) -> bool {
            self.running.load(Ordering::Relaxed)
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
                .name("loss")
                .marker(symbols::Marker::Dot)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(Color::Cyan))
                .data(data),
        ];

        let chart = Chart::new(datasets)
            .block(Block::bordered().title("Loss"))
            .x_axis(
                Axis::default()
                    .title("Step")
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
                    .title("Loss")
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

    impl Drop for Monitor {
        fn drop(&mut self) {
            // signal the thread to stop
            self.running.store(false, Ordering::Relaxed);
            // wait for it to fully finish and call ratatui::restore()
            if let Some(handle) = self.thread.take() {
                let _ = handle.join();
            }
        }
    }
}
