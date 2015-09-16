#![allow(non_snake_case, non_upper_case_globals)]
// #![cfg_attr(test, feature(test))]
extern crate time;

use time::precise_time_s;
use std::mem;

///////////////////////////////////////////////////////////////////////////////
// 1. Calibration
///////////////////////////////////////////////////////////////////////////////
const aalpha:f64 = 0.33333333333;  // Elasticity of output w.r.t. capital
const bbeta:f64 = 0.95;           // Discount factor

// Productivity values
const vProductivity:[f64; 5] = [0.9792, 0.9896, 1.0000, 1.0106, 1.0212];

// Transition matrix
const mTransition:[[f64; 5]; 5] = [[0.9727, 0.0273, 0.0000, 0.0000, 0.0000],
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
    let mut mValueFunction = vec![[0f64; nGridProductivity]; nGridCapital];
    let mut mValueFunctionNew = mValueFunction.clone();
    let mut mPolicyFunction = mValueFunction.clone();
    let mut expectedValueFunction = mValueFunction.clone();

    // 4. We pre-build output for each point in the grid
    let mOutput = (0..nGridCapital)
                      .map(|nCapital| {
                          (0..nGridProductivity).map(|nProductivity|
        vProductivity[nProductivity]*vGridCapital[nCapital].powf(aalpha)
      ).collect::<Vec<f64>>()
                      })
                      .collect::<Vec<Vec<f64>>>();

  // 5. Main iteration
  // TODO: one could implement a macro for the multiple declarations
    let mut maxDifference = 10_f64;
    const tolerance: f64 = 0.0000001; // compiler warn: variable does not need to be mutable
    let mut capitalChoice: f64;

    let mut iteration = 0;

    while maxDifference > tolerance {
        for nProductivity in 0..nGridProductivity {
            let transitions = &mTransition[nProductivity];

            for nCapital in 0..nGridCapital {
                let value_fns = &mValueFunction[nCapital];

                let expected_value = transitions.iter()
                                                .zip(value_fns.iter())
                                                .fold(0.0f64, |acc, (transition, value_fn)| {
                    acc + (transition * value_fn)
                });

                expectedValueFunction[nCapital][nProductivity] = expected_value;
            }
        }

        for nProductivity in 0..nGridProductivity {

        // We start from previous choice (monotonicity of policy function)
            let mut gridCapitalNextPeriod = 0;

            for nCapital in 0..nGridCapital {

                let mut valueHighSoFar = -100000.0;

                for nCapitalNextPeriod in gridCapitalNextPeriod..nGridCapital {

                    let consumption = mOutput[nCapital][nProductivity] -
                                  vGridCapital[nCapitalNextPeriod];
                    let valueProvisional = (1_f64 - bbeta) * (consumption.ln()) +
                                       bbeta *
                                       expectedValueFunction[nCapitalNextPeriod][nProductivity];

                    if valueProvisional > valueHighSoFar {
                        valueHighSoFar = valueProvisional;
                        capitalChoice = vGridCapital[nCapitalNextPeriod];
                        gridCapitalNextPeriod = nCapitalNextPeriod;
                    } else {
                        break; // We break when we have achieved the max
                    }

                    mValueFunctionNew[nCapital][nProductivity] = valueHighSoFar;
                    mPolicyFunction[nCapital][nProductivity] = capitalChoice;
                }

            }

        }

        {
            let old_vals = mValueFunction.iter().flat_map(|arr| arr.iter());
            let new_vals = mValueFunctionNew.iter().flat_map(|arr| arr.iter());

            maxDifference = -100000.0;
            for diff in old_vals.zip(new_vals).map(|(old, new)| (old - new).abs()) {
                if diff > maxDifference {
                    maxDifference = diff
                }
            }
        }

        // swap buffers after the loop
        mem::swap(&mut mValueFunction, &mut mValueFunctionNew);

        iteration += 1;
        if print && (iteration % 10 == 0 || iteration == 1) {
            println!("Iteration = {}, Sup Diff = {}", iteration, maxDifference);
        }
    }

    if print {
        println!("Iteration = {}, Sup Diff = {}", iteration, maxDifference);
    }

    mPolicyFunction[999][2]
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
