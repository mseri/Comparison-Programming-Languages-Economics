#![allow(non_snake_case, non_upper_case_globals)]
// #![cfg_attr(test, feature(test))]
extern crate time;
extern crate num_cpus;
extern crate scoped_threadpool;

use time::precise_time_s;
use scoped_threadpool::Pool;
use std::mem;

///////////////////////////////////////////////////////////////////////////////
// 1. Calibration
///////////////////////////////////////////////////////////////////////////////
const aalpha:f64 = 0.33333333333;  // Elasticity of output w.r.t. capital
const bbeta:f64 = 0.95;           // Discount factor

// Productivity values
const vProductivity:[f64; 5] = [0.9792, 0.9896, 1.0000, 1.0106, 1.0212];

// Transition matrix
static mTransition:[[f64; 5]; 5] = [[0.9727, 0.0273, 0.0000, 0.0000, 0.0000],
                                    [0.0041, 0.9806, 0.0153, 0.0000, 0.0000],
                                    [0.0000, 0.0082, 0.9837, 0.0082, 0.0000],
                                    [0.0000, 0.0000, 0.0153, 0.9806, 0.0041],
                                    [0.0000, 0.0000, 0.0000, 0.0273, 0.9727]];

// Dimensions to generate the grid of capital
const nGridCapital: usize = 17820;
const nGridProductivity: usize = 5;

fn solve(print: bool) -> f64 {
  /////////////////////////////////////////////////////////////////////////////
  // 2. Steady State
  /////////////////////////////////////////////////////////////////////////////

    let thread_count = num_cpus::get();
    let mut pool = Pool::new(thread_count as u32);

    let capitalSteadyState: f64 = (aalpha * bbeta).powf(1_f64 / (1_f64 - aalpha));
    let outputSteadyState: f64 = capitalSteadyState.powf(aalpha);
    let consumptionSteadyState: f64 = outputSteadyState - capitalSteadyState;

    if print {
        println!("Output = {}, Capital = {}, Consumption = {}",
                 outputSteadyState,
                 capitalSteadyState,
                 consumptionSteadyState);
    }

    let mut vGridCapital = [0f64; nGridCapital];

    for (i, val) in vGridCapital.iter_mut().enumerate() {
        *val = 0.5 * capitalSteadyState + 0.00001 * (i as f64)
    }

    // 3. Required matrices and vectors
    let mut mValueFunction: [Vec<f64>; nGridProductivity] = [
        vec![0f64; nGridCapital],
        vec![0f64; nGridCapital],
        vec![0f64; nGridCapital],
        vec![0f64; nGridCapital],
        vec![0f64; nGridCapital],
    ];

    let mut mPolicyFunction: [Vec<f64>; nGridProductivity] = [
        vec![0f64; nGridCapital],
        vec![0f64; nGridCapital],
        vec![0f64; nGridCapital],
        vec![0f64; nGridCapital],
        vec![0f64; nGridCapital],
    ];

    let mut expectedValueFunction: [Vec<f64>; nGridProductivity] = [
        vec![0f64; nGridCapital],
        vec![0f64; nGridCapital],
        vec![0f64; nGridCapital],
        vec![0f64; nGridCapital],
        vec![0f64; nGridCapital],
    ];

    // 4. We pre-build output for each point in the grid
    let mOutput : Vec<[f64; nGridProductivity]> = (0..nGridCapital).map(|nCapital| {
        let mut arr = [0.0; nGridProductivity];

        for (nProductivity, slot) in arr.iter_mut().enumerate() {
            *slot = vProductivity[nProductivity] * vGridCapital[nCapital].powf(aalpha)
        }

        arr
    }).collect();

  // 5. Main iteration
  // TODO: one could implement a macro for the multiple declarations
    let mut maxDifference = 10_f64;
    const tolerance: f64 = 0.0000001; // compiler warn: variable does not need to be mutable

    let mut iteration = 0;

    // small array to split where the writes are going, so we don't need a mutex
    let mut differences = [-100000.0; nGridProductivity];

    while maxDifference > tolerance {
        pool.scoped(|scoped| {
            for (expected_values, transitions) in expectedValueFunction.iter_mut()
                                                                       .zip(mTransition.iter()) {
                // Only capture refs
                let mValueFunction = &mValueFunction;

                scoped.execute(move || {
                    for (idx, expected_value) in expected_values.iter_mut().enumerate() {
                        *expected_value = transitions.iter()
                                                     .zip(mValueFunction.iter())
                                                     .fold(0.0f64, |acc, (transition, value_fns)| {
                            acc + (transition * value_fns[idx])
                        });
                    }
                });
            }
        });

        pool.scoped(|scoped| {
            for (nProductivity, (((policies, value_fns), maxDifference), expected_values)) in mPolicyFunction.iter_mut()
                                                                         .zip(mValueFunction.iter_mut())
                                                                         .zip(differences.iter_mut())
                                                                         .zip(expectedValueFunction.iter())
                                                                         .enumerate() {
                *maxDifference = -100000.0;

                // Only capture refs
                let mOutput = &mOutput;

                scoped.execute(move || {
                    // We start from previous choice (monotonicity of policy function)
                    let mut gridCapitalNextPeriod = 0;

                    for ((policy, output), value_fn) in policies.iter_mut()
                                                                .zip(mOutput.iter())
                                                                .zip(value_fns.iter_mut()) {
                        let mut valueHighSoFar = -100000.0;
                        let mut capitalChoice = vGridCapital[0];
                        let mOutput_cache = &output[nProductivity];

                        for nCapitalNextPeriod in gridCapitalNextPeriod..nGridCapital {
                            let consumption = mOutput_cache - &vGridCapital[nCapitalNextPeriod];
                            let valueProvisional = (1_f64 - bbeta) * (consumption.ln()) +
                                                    bbeta *
                                                    expected_values[nCapitalNextPeriod];

                            if valueProvisional > valueHighSoFar {
                                valueHighSoFar = valueProvisional;
                                capitalChoice = vGridCapital[nCapitalNextPeriod];
                                gridCapitalNextPeriod = nCapitalNextPeriod;
                            } else {
                                break; // We break when we have achieved the max
                            }
                        }

                        let old = mem::replace(value_fn, valueHighSoFar);
                        let diff = (old - valueHighSoFar).abs();
                        if diff > *maxDifference {
                            *maxDifference = diff
                        }

                        *policy = capitalChoice;
                    }
                })
            }
        });

        maxDifference = -100000.0;
        for &diff in &differences {
            if diff > maxDifference {
                maxDifference = diff
            }
        }

        iteration += 1;
        if print && (iteration % 10 == 0 || iteration == 1) {
            println!("Iteration = {}, Sup Diff = {}", iteration, maxDifference);
        }
    }

    if print {
        println!("Iteration = {}, Sup Diff = {}", iteration, maxDifference);
    }

    mPolicyFunction[2][999]
}

fn main() {
    let sample = false;

    if sample {
        let mut samples = (0..5).map(|i| {
            let cpu0 = precise_time_s();
            let result = solve(false);
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
        let result = solve(true);
        let cpu1 = precise_time_s();

        println!("My check = {}\n", result);
        assert_eq!(result, 0.1465491436956954);
        println!("Elapsed time is = {}", cpu1 - cpu0);
    }
}


// FIXME: Can't use bench as the loop is too slow (takes 8mins to get samples)
// #[cfg(test)]
// mod test {
//     extern crate test;
//     use self::test::Bencher;
//     use super::solve;

//     #[bench]
//     fn bench(b: &mut Bencher) {
//         b.iter(|| {
//             // use `test::black_box` to prevent compiler optimizations from disregarding
//             // unused values
//             test::black_box(solve(false));
//         });
//     }
// }
