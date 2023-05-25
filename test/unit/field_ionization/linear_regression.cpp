#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>

// Structure to hold the data points
struct DataPoint {
    double x;
    double y;
};

// Function to fit a line to the data using linear regression
void fitLineToData(const std::vector<DataPoint>& data, double& slope, double& intercept, double& stdErrorSlope, double& stdErrorIntercept, double& fitUncertainty) {
    double sumX = 0.0;
    double sumY = 0.0;
    double sumXY = 0.0;
    double sumXX = 0.0;
    int n = data.size();

    // Calculate the required sums
    for (const auto& point : data) {
        sumX += point.x;
        sumY += point.y;
        sumXY += (point.x * point.y);
        sumXX += (point.x * point.x);
    }

    // Calculate the slope and intercept
    double denominator = n * sumXX - sumX * sumX;
    if (denominator != 0) {
        slope = (n * sumXY - sumX * sumY) / denominator;
        intercept = (sumY - slope * sumX) / n;
    } else {
        // Handle the case when the denominator is zero (e.g., division by zero)
        slope = 0.0;
        intercept = 0.0;
    }

    // Calculate the standard error of the slope
    double residualSumOfSquares = 0.0;
    for (const auto& point : data) {
        double error = point.y - (slope * point.x + intercept);
        residualSumOfSquares += (error * error);
    }
    double meanSquaredError = residualSumOfSquares / (n - 2);
    fitUncertainty = std::sqrt(meanSquaredError);
    stdErrorSlope = std::sqrt(meanSquaredError / denominator);

    // Calculate the standard error of the intercept
    double meanX = sumX / n;
    stdErrorIntercept = std::sqrt(meanSquaredError * (1.0 / n + meanX * meanX / denominator));

}

// Function to test if the simulation passes
bool fitAgreement(const std::vector<DataPoint>& data, double slopeAnalytic, double interceptAnalytic, double slopeFit, double interceptFit, double fitUncertainty) {
  bool flag = 1; // flag that decides if test passes
  double N_sigma = 2;  // How many sigma uncertainty
  
  for (const auto& point : data) {
    // Calculate y values of the fit and analytic
    double yFit = slopeFit * point.x + interceptFit ;
    double yAnalytic = slopeAnalytic * point.x + interceptAnalytic;

    // Calculate bounds on the fit
    double fitLowerBound = yFit - N_sigma*fitUncertainty;
    double fitUpperBound = yFit + N_sigma*fitUncertainty;

    // Check of the analytic is outside the bounds
    if (yAnalytic > fitUpperBound || yAnalytic < fitLowerBound) {
      flag = 0;
    }
  } //for loop

  return flag;
}



















