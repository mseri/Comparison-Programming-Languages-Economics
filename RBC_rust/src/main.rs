#![allow(non_snake_case, non_upper_case_globals)]
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

fn main() {

    let cpu0 = precise_time_s();

  /////////////////////////////////////////////////////////////////////////////
  // 2. Steady State
  /////////////////////////////////////////////////////////////////////////////

    let capitalSteadyState: f64 = (aalpha * bbeta).powf(1_f64 / (1_f64 - aalpha));
    let outputSteadyState: f64 = capitalSteadyState.powf(aalpha);
    let consumptionSteadyState: f64 = outputSteadyState - capitalSteadyState;

    println!("\
    Output = {}, Capital = {}, Consumption = {}",
    outputSteadyState, capitalSteadyState, consumptionSteadyState);

    let vGridCapital = (0..nGridCapital)
                           .map(|nCapital| 0.5 * capitalSteadyState + 0.00001 * (nCapital as f64))
                           .collect::<Vec<f64>>();

  // 3. Required matrices and vectors
  // One could use: vec![vec![0_f64; nGridProductivity]; nGridCapital];
  // but for some reasons this is faster.

    #[inline]
    fn row() -> Vec<f64> {
        (0..nGridProductivity).map(|_| 0_f64).collect::<Vec<f64>>()
    }

    let mut mValueFunction = (0..nGridCapital).map(|_| row()).collect::<Vec<Vec<f64>>>();
    let mut mValueFunctionNew = (0..nGridCapital).map(|_| row()).collect::<Vec<Vec<f64>>>();
    let mut mPolicyFunction = (0..nGridCapital).map(|_| row()).collect::<Vec<Vec<f64>>>();
    let mut expectedValueFunction = (0..nGridCapital).map(|_| row()).collect::<Vec<Vec<f64>>>();

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
    let mut diff: f64;
    let tolerance = 0.0000001_f64; // compiler warn: variable does not need to be mutable
    let mut valueHighSoFar: f64;
    let mut valueProvisional: f64;
    let mut consumption: f64;
    let mut capitalChoice: f64;

    let mut iteration = 0;

    while maxDifference > tolerance {

        for nProductivity in 0..nGridProductivity {
            for nCapital in 0..nGridCapital {
                expectedValueFunction[nCapital][nProductivity] = 0.0;
                for nProductivityNextPeriod in 0..nGridProductivity {
                    expectedValueFunction[nCapital][nProductivity] +=
                        mTransition[nProductivity][nProductivityNextPeriod] *
                        mValueFunction[nCapital][nProductivityNextPeriod];
                }
            }
        }

        for nProductivity in 0..nGridProductivity {

        // We start from previous choice (monotonicity of policy function)
            let mut gridCapitalNextPeriod = 0;

            for nCapital in 0..nGridCapital {

                valueHighSoFar = -100000.0;
            //capitalChoice  = vGridCapital[0]; // compiler warn: is never read

                for nCapitalNextPeriod in gridCapitalNextPeriod..nGridCapital {

                    consumption = mOutput[nCapital][nProductivity] -
                                  vGridCapital[nCapitalNextPeriod];
                    valueProvisional = (1_f64 - bbeta) * (consumption.ln()) +
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

        let mut diffHighSoFar = -100000.0;
        for nProductivity in 0..nGridProductivity {
            for nCapital in 0..nGridCapital {
                diff = (mValueFunction[nCapital][nProductivity] -
                        mValueFunctionNew[nCapital][nProductivity])
                           .abs();
                if diff > diffHighSoFar {
                    diffHighSoFar = diff;
                }
            }
        }

        // swap buffers after the loop
        mem::swap(&mut mValueFunction, &mut mValueFunctionNew);

        maxDifference = diffHighSoFar;

        iteration += 1;
        if iteration % 10 == 0 || iteration == 1 {
            println!("Iteration = {}, Sup Diff = {}", iteration, maxDifference);
        }
    }

    println!("Iteration = {}, Sup Diff = {}\n", iteration, maxDifference);
    println!("My check = {}\n", mPolicyFunction[999][2]);

    let cpu1 = precise_time_s();

    println!("Elapsed time is   = {}", cpu1  - cpu0);
}
