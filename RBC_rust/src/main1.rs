#![allow(non_snake_case, non_upper_case_globals)]
extern crate time;
use time::precise_time_s;

///////////////////////////////////////////////////////////////////////////////
// 1. Calibration
///////////////////////////////////////////////////////////////////////////////
static aalpha:f64 = 1_f64/3_f64;  // Elasticity of output w.r.t. capital
static bbeta:f64 = 0.95;           // Discount factor

// Productivity values
const vProductivity:[f64; 5] = [0.9792, 0.9896, 1.0000, 1.0106, 1.0212];

// Transition matrix
const mTransition:[[f64; 5]; 5] = [
    [0.9727, 0.0273, 0.0000, 0.0000, 0.0000],
    [0.0041, 0.9806, 0.0153, 0.0000, 0.0000],
    [0.0000, 0.0082, 0.9837, 0.0082, 0.0000],
    [0.0000, 0.0000, 0.0153, 0.9806, 0.0041],
    [0.0000, 0.0000, 0.0000, 0.0273, 0.9727]
];

// Dimensions to generate the grid of capital
const nGridCapital: usize = 17820;
const nGridProductivity: usize = 5;

fn main() {

  let cpu0 = precise_time_s();

  /////////////////////////////////////////////////////////////////////////////
  // 2. Steady State
  /////////////////////////////////////////////////////////////////////////////

  let capitalSteadyState:f64 = (aalpha*bbeta).powf(1_f64/(1_f64-aalpha));
  let outputSteadyState:f64  = capitalSteadyState.powf(aalpha);
  let consumptionSteadyState:f64 = outputSteadyState-capitalSteadyState;

  println!("\
    Output = {}, Capital = {}, Consumption = {}", 
    outputSteadyState, capitalSteadyState, consumptionSteadyState);

  let vGridCapital = (0..nGridCapital).map(|nCapital| 
    0.5*capitalSteadyState + 0.00001*(nCapital as f64)
  ).collect::<Vec<f64>>();

  // 3. Required matrices and vectors

  let mut mValueFunction = Box::new([[0f64; nGridProductivity]; nGridCapital]);
  let mut mValueFunctionNew = Box::new([[0f64; nGridProductivity]; nGridCapital]);
  let mut mPolicyFunction = Box::new([[0f64; nGridProductivity]; nGridCapital]);
  let mut expectedValueFunction = Box::new([[0f64; nGridProductivity]; nGridCapital]);

  // 4. We pre-build output for each point in the grid

  let mut mOutput = Box::new([[0f64; nGridProductivity]; nGridCapital]);
  for nProductivity in 0..nGridProductivity {
    for nCapital in 0..nGridCapital {
      unsafe {
        mOutput[nCapital][nProductivity] =
        vProductivity.get_unchecked(nProductivity)*vGridCapital.get_unchecked(nCapital).powf(aalpha);
      }
    }
  }
  let mOutput = mOutput;

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
              unsafe {
                expectedValueFunction[nCapital][nProductivity] += mTransition.get_unchecked(nProductivity).get_unchecked(nProductivityNextPeriod)*mValueFunction.get_unchecked(nCapital).get_unchecked(nProductivityNextPeriod);
              }
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
                unsafe {
                  consumption = mOutput.get_unchecked(nCapital).get_unchecked(nProductivity)-vGridCapital.get_unchecked(nCapitalNextPeriod);
                  valueProvisional = (1_f64-bbeta)*(consumption.ln())+bbeta*expectedValueFunction.get_unchecked(nCapitalNextPeriod).get_unchecked(nProductivity);
                }
                
                if valueProvisional>valueHighSoFar {
                    valueHighSoFar = valueProvisional;
                    unsafe {
                      capitalChoice = *vGridCapital.get_unchecked(nCapitalNextPeriod);
                    }
                    gridCapitalNextPeriod = nCapitalNextPeriod;
                }
                else{
                    break; // We break when we have achieved the max
                }
                
                mValueFunctionNew[nCapital][nProductivity] = valueHighSoFar;
                mPolicyFunction[nCapital][nProductivity] = capitalChoice;
            }
            
        }
        
    }
    
    diff = -100000.0;
    for nProductivity in 0..nGridProductivity {
        for nCapital in 0..nGridCapital {
          unsafe {
            let newVal = mValueFunctionNew.get_unchecked(nCapital).get_unchecked(nProductivity);
            diff = diff.max((mValueFunction.get_unchecked(nCapital).get_unchecked(nProductivity)-newVal).abs());
            mValueFunction[nCapital][nProductivity] = *newVal;
          }
        }
    }
    maxDifference = diff;
    
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
