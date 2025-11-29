// clearance_tests.scad
// This file contains test scripts to verify the clearance and fit of the SIM cards within the designed tubes.

include <../models/sim_tube_nano.scad>
include <../models/sim_tube_micro.scad>
include <../models/sim_tube_mini.scad>
include <../types/dimensions.scad>

// Test for Nano-SIM tube clearance
module test_nano_sim_clearance() {
    echo("Testing Nano-SIM Tube Clearance");
    sim_tube_nano();
}

// Test for Micro-SIM tube clearance
module test_micro_sim_clearance() {
    echo("Testing Micro-SIM Tube Clearance");
    sim_tube_micro();
}

// Test for Mini-SIM tube clearance
module test_mini_sim_clearance() {
    echo("Testing Mini-SIM Tube Clearance");
    sim_tube_mini();
}

// Run all clearance tests
test_nano_sim_clearance();
test_micro_sim_clearance();
test_mini_sim_clearance();