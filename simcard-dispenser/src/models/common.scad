// common.scad
// This file includes common functions and parameters for SIM card tube models.

module sim_card_dimensions() {
    // Dimensions for different SIM card types
    return [
        ["Full-Size SIM", 85.6, 53.98, 0.76],
        ["Mini-SIM", 25, 15, 0.76],
        ["Micro-SIM", 15, 12, 0.76],
        ["Nano-SIM", 12.3, 8.8, 0.67]
    ];
}

function get_sim_card_dimensions(type) = 
    sim_card_dimensions()[type];

// Common parameters
clearance = 0.2; // Clearance for card movement
slot_height = 0.8; // Height of the slot for dispensing cards
pusher_thickness = 0.5; // Thickness of the pusher mechanism

// Function to create a pusher
module create_pusher(width, length) {
    translate([0, 0, 0])
        cube([width, pusher_thickness, length], center=false);
}

// Function to create a slot
module create_slot(width, height) {
    translate([0, 0, 0])
        cube([width, 1, height], center=false);
}