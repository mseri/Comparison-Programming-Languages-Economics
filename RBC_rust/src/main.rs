extern crate time;
extern crate num_cpus;
extern crate scoped_threadpool;

use time::precise_time_s;
use scoped_threadpool::Pool;
use std::mem;

// 1. Calibration
const ALPHA: f64 = 0.33333333333;  // Elasticity of output w.r.t. capital
const BETA: f64 = 0.95;           // Discount factor

// Productivity values
const PRODUCTIVITY: [f64; 5] = [0.9792, 0.9896, 1.0000, 1.0106, 1.0212];

// Transition matrix
static TRANSITIONS: [[f64; 5]; 5] = [[0.9727, 0.0273, 0.0000, 0.0000, 0.0000],
                                     [0.0041, 0.9806, 0.0153, 0.0000, 0.0000],
                                     [0.0000, 0.0082, 0.9837, 0.0082, 0.0000],
                                     [0.0000, 0.0000, 0.0153, 0.9806, 0.0041],
                                     [0.0000, 0.0000, 0.0000, 0.0273, 0.9727]];

// Dimensions to generate the grid of capital
const GRID_CAPITAL: usize = 17820;
const GRID_PRODUCTIVITY: usize = 5;

fn solve(pool: &mut Pool, print: bool) -> f64 {
    // 2. Steady State
    let capital_steady_state = (ALPHA * BETA).powf(1_f64 / (1_f64 - ALPHA));
    let output_steady_state = capital_steady_state.powf(ALPHA);
    let consumption_steady_state = output_steady_state - capital_steady_state;

    if print {
        println!("Output = {}, Capital = {}, Consumption = {}",
                 output_steady_state,
                 capital_steady_state,
                 consumption_steady_state);
    }

    let mut grid_capital = [0f64; GRID_CAPITAL];

    for (i, val) in grid_capital.iter_mut().enumerate() {
        *val = 0.5 * capital_steady_state + 0.00001 * (i as f64)
    }

    // 3. Required matrices and vectors
    let mut values: [Vec<f64>; GRID_PRODUCTIVITY] = [
        vec![0f64; GRID_CAPITAL],
        vec![0f64; GRID_CAPITAL],
        vec![0f64; GRID_CAPITAL],
        vec![0f64; GRID_CAPITAL],
        vec![0f64; GRID_CAPITAL],
    ];

    let mut policies: [Vec<f64>; GRID_PRODUCTIVITY] = [
        vec![0f64; GRID_CAPITAL],
        vec![0f64; GRID_CAPITAL],
        vec![0f64; GRID_CAPITAL],
        vec![0f64; GRID_CAPITAL],
        vec![0f64; GRID_CAPITAL],
    ];

    let mut expected_values: [Vec<f64>; GRID_PRODUCTIVITY] = [
        vec![0f64; GRID_CAPITAL],
        vec![0f64; GRID_CAPITAL],
        vec![0f64; GRID_CAPITAL],
        vec![0f64; GRID_CAPITAL],
        vec![0f64; GRID_CAPITAL],
    ];

    // 4. We pre-build output for each point in the grid
    let outputs: [Vec<f64>; GRID_PRODUCTIVITY] = [
        (0..GRID_CAPITAL).map(|capital_idx| PRODUCTIVITY[0] * grid_capital[capital_idx].powf(ALPHA)).collect(),
        (0..GRID_CAPITAL).map(|capital_idx| PRODUCTIVITY[1] * grid_capital[capital_idx].powf(ALPHA)).collect(),
        (0..GRID_CAPITAL).map(|capital_idx| PRODUCTIVITY[2] * grid_capital[capital_idx].powf(ALPHA)).collect(),
        (0..GRID_CAPITAL).map(|capital_idx| PRODUCTIVITY[3] * grid_capital[capital_idx].powf(ALPHA)).collect(),
        (0..GRID_CAPITAL).map(|capital_idx| PRODUCTIVITY[4] * grid_capital[capital_idx].powf(ALPHA)).collect(),
    ];

    // 5. Main iteration
    const TOLERANCE: f64 = 0.0000001; // compiler warn: variable does not need to be mutable

    let mut iteration = 0;
    let mut finished = false;

    // small array to split where the writes are going, so we don't need a mutex
    let mut diffs = [-100000.0; GRID_PRODUCTIVITY];

    while !finished {
        pool.scoped(|scoped| {
            for (expected_values, transitions) in expected_values.iter_mut()
                                                                 .zip(TRANSITIONS.iter()) {
                // Only capture refs
                let values = &values;

                scoped.execute(move || {
                    for (idx, expected_value) in expected_values.iter_mut().enumerate() {
                        *expected_value = transitions.iter()
                                                     .zip(values.iter())
                                                     .fold(0.0f64, |acc, (transition, value_fns)| {
                            acc + (transition * value_fns[idx])
                        });
                    }
                });
            }
        });

        pool.scoped(|scoped| {
            for ((((policies, value_fns), max_diff), expected_values), outputs) in policies.iter_mut()
                                                                                           .zip(values.iter_mut())
                                                                                           .zip(diffs.iter_mut())
                                                                                           .zip(expected_values.iter())
                                                                                           .zip(outputs.iter()) {
                scoped.execute(move || {
                    *max_diff = -100000.0;

                    // We start from previous choice (monotonicity of policy function)
                    let mut grid_capital_next_period = 0;

                    for ((policy, output), value_fn) in policies.iter_mut()
                                                                .zip(outputs.iter())
                                                                .zip(value_fns.iter_mut()) {
                        let mut value_high = -100000.0;
                        let mut capital_choice = grid_capital[0];

                        for capital_next_period in grid_capital_next_period..GRID_CAPITAL {
                            let consumption = output - &grid_capital[capital_next_period];
                            let value_provisional = (1.0 - BETA) * (consumption.ln()) +
                                                    BETA *
                                                    expected_values[capital_next_period];

                            if value_provisional > value_high {
                                value_high = value_provisional;
                                capital_choice = grid_capital[capital_next_period];
                                grid_capital_next_period = capital_next_period;
                            } else {
                                break; // We break when we have achieved the max
                            }
                        }

                        let old = mem::replace(value_fn, value_high);
                        let diff = (old - value_high).abs();
                        if diff > *max_diff {
                            *max_diff = diff
                        }

                        *policy = capital_choice;
                    }
                })
            }
        });

        let mut max_diff = diffs[0];
        for &diff in &diffs[1..] {
            if diff > max_diff {
                max_diff = diff
            }
        }

        finished = max_diff <= TOLERANCE;

        iteration += 1;
        if print && (iteration % 10 == 0 || iteration == 1 || finished) {
            println!("Iteration = {}, Sup Diff = {}", iteration, max_diff);
        }
    }

    policies[2][999]
}

fn main() {
    use std::env::args;

    let sample = args().any(|arg| arg == "--sample");
    let thread_count = num_cpus::get();
    let ref mut pool = Pool::new(thread_count as u32);

    if sample {
        let mut samples = (0..5).map(|i| {
            let cpu0 = precise_time_s();
            let result = solve(pool, false);
            let cpu1 = precise_time_s();

            assert_eq!(result, 0.1465491436956954);

            let diff = cpu1 - cpu0;
            println!("Sample #{}, Time: {}s", i + 1, diff);
            diff
        }).collect::<Vec<f64>>();

        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("\nMedian time is = {}", samples[2]);
    } else {
        let cpu0 = precise_time_s();
        let result = solve(pool, true);
        let cpu1 = precise_time_s();

        println!("My check = {}\n", result);
        assert_eq!(result, 0.1465491436956954);
        println!("Elapsed time is = {}", cpu1 - cpu0);
    }
}
