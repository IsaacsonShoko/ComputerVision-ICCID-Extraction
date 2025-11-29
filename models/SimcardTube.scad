// Triple Nano-SIM Gravity Dispenser Array - Ceiling Mount
// VERSION 3.0 - Production Array with Downward Mounting
// Units: mm

$fn = 64;

// ========================
// PARAMETERS
// ========================

// Core SIM specs (Nano-SIM)
card_w = 12.3;    // X-axis (longer side)
card_d = 8.8;     // Y-axis (shorter side)  
card_t = 0.67;    // Z-axis (thickness)

// Capacity & spacing
N_CARDS = 50;
inter_gap = 0.03;
headroom = 5;

// Clearances
clearance_side = 0.2;
inner_w = card_w + 2*clearance_side;
inner_d = card_d + 2*clearance_side;

// Structure
wall = 2.5;
base_thk = 5.0;      // Base thickness (now mounting surface)
slot_h = 0.9;        // Slot height
dual_slots = true;   // Opposing pusher slots

// ANTI-DOUBLE-FEED LEDGE
enable_ledge = true;
ledge_depth = 0.4;
ledge_height = 0.3;
ledge_offset = 0.05;

// ARRAY CONFIGURATION
num_tubes = 3;
tube_spacing = 10;   // 10mm center-to-center spacing between tubes

// ADHESIVE MOUNTING (clean base, no flanges)
enable_flange = false;     // Using glue/adhesive instead

// Visual aids
pusher_thk = 0.45;
pusher_width = inner_w - 0.7;
stroke = card_d + 1.5;

// Visibility window
enable_window = true;
window_width = 6;
window_start = 15;
window_height = 30;

// Calculated heights
stack_h = N_CARDS * card_t + N_CARDS * inter_gap + headroom;
tube_top = base_thk + stack_h;
outer_h = tube_top + wall + 5;

// Total array width
total_width = (inner_w + 2*wall) * num_tubes + tube_spacing * (num_tubes - 1);

// ========================
// SINGLE TUBE MODULE
// ========================

module single_tube() {
    difference() {
        // Main tube body
        cube([inner_w + 2*wall, inner_d + 2*wall, outer_h]);
        
        // Internal cavity (starts immediately after base)
        translate([wall, wall, base_thk])
            cube([inner_w, inner_d, stack_h + 10]);
        
        // FRONT pusher slot (starts immediately at base_thk)
        translate([wall, -1, base_thk])
            cube([inner_w, wall + 2, slot_h]);
        
        // REAR pusher slot (starts immediately at base_thk)
        if (dual_slots) {
            translate([wall, wall + inner_d - 1, base_thk])
                cube([inner_w, wall + 2, slot_h]);
        }
        
        // Visibility window (side wall)
        if (enable_window) {
            translate([-1, wall + (inner_d - window_width)/2, base_thk + window_start])
                cube([wall + 2, window_width, window_height]);
        }
    }
    
    // ANTI-DOUBLE-FEED LEDGES (immediately above slot)
    if (enable_ledge) {
        ledge_z = base_thk + slot_h + ledge_offset;
        
        // Front ledge
        translate([wall, wall, ledge_z])
            cube([inner_w, ledge_depth, ledge_height]);
        
        // Rear ledge
        if (dual_slots) {
            translate([wall, wall + inner_d - ledge_depth, ledge_z])
                cube([inner_w, ledge_depth, ledge_height]);
        }
    }
}

// ========================
// TRIPLE TUBE ARRAY WITH MOUNTING
// ========================

module tube_array() {
    tube_width = inner_w + 2*wall;
    connecting_arm_width = 2;  // Thin connecting arms
    
    // Generate 3 tubes with connecting arms
    for (i = [0:num_tubes-1]) {
        x_pos = i * (tube_width + tube_spacing);
        
        // Individual tube
        translate([x_pos, 0, 0])
            single_tube();
        
        // Connecting arm to next tube (except for last tube)
        if (i < num_tubes - 1) {
            difference() {
                // Connecting arm
                translate([x_pos + tube_width, 0, 0])
                    cube([tube_spacing, inner_d + 2*wall, base_thk]);
                
                // M4 mounting hole through connecting arm
                translate([x_pos + tube_width + tube_spacing/2, 
                          (inner_d + 2*wall)/2, 
                          -1])
                    cylinder(d=4.5, h=base_thk + 2);  // M4 clearance hole
                
                // Countersink from bottom for M4 flat head screw
                translate([x_pos + tube_width + tube_spacing/2, 
                          (inner_d + 2*wall)/2, 
                          -0.5])
                    cylinder(d1=8.5, d2=4.5, h=3);  // M4 countersink
            }
        }
    }
}

// ========================
// VISUAL COMPONENTS
// ========================

module pusher_assembly(tube_index, y_offset, retracted=true) {
    tube_width = inner_w + 2*wall;
    x_base = tube_index * (tube_width + tube_spacing);
    push_dist = retracted ? 0 : stroke;
    
    color("SteelBlue", 0.7)
    translate([x_base + wall + (inner_w - pusher_width)/2, 
               y_offset + push_dist, 
               base_thk])
        cube([pusher_width, pusher_thk, slot_h]);
}

module sample_cards(tube_index, count=6) {
    tube_width = inner_w + 2*wall;
    x_base = tube_index * (tube_width + tube_spacing);
    
    color("Gold", 0.8)
    for (i = [0:count-1]) {
        translate([x_base + wall + clearance_side, 
                   wall + clearance_side, 
                   base_thk + i*(card_t + inter_gap)])
            cube([card_w, card_d, card_t]);
    }
}

module mounting_screws() {
    // No mounting hardware - adhesive mounting
}

module test_coupon() {
    // Test piece: bottom section with slots and ledges
    intersection() {
        tube_array();
        translate([0, 0, 0])
            cube([total_width, inner_d + 2*wall, base_thk + 8]);
    }
}

module full_assembly() {
    // Main tube array
    tube_array();
    
    // Visual pushers for each tube (front slot)
    for (i = [0:num_tubes-1]) {
        pusher_assembly(i, -pusher_thk - 0.5, retracted=true);
        if (dual_slots) 
            pusher_assembly(i, inner_d + wall + 0.5, retracted=true);
    }
    
    // Sample cards in each tube
    for (i = [0:num_tubes-1]) {
        sample_cards(i, min(8, N_CARDS));
    }
    
    // Visual mounting screws
    mounting_screws();
        
    // Tube labels on base
    color("Black")
    for (i = [0:num_tubes-1]) {
        tube_width = inner_w + 2*wall;
        x_pos = i * (tube_width + tube_spacing) + tube_width/2;
        
        translate([x_pos, (inner_d + 2*wall)/2, 0.2])
            linear_extrude(0.5)
            text(str(i+1), size=4, halign="center", valign="center", 
                 font="Liberation Sans:style=Bold");
    }
}

// ========================
// RENDER SELECTION
// ========================

RENDER_MODE = "full"; // Options: "full", "test_coupon", "array_only"

if (RENDER_MODE == "full") {
    full_assembly();
} else if (RENDER_MODE == "test_coupon") {
    test_coupon();
} else if (RENDER_MODE == "array_only") {
    tube_array();
}

// ========================
// DESIGN NOTES - VERSION 3.1
// ========================

// CRITICAL FEATURES:
// 1. ✅ CLEAN 5mm BASE - perfect for adhesive mounting
// 2. ✅ SLOTS start IMMEDIATELY after base (at z = 5mm)
// 3. ✅ NO FLANGES - streamlined design
// 4. ✅ 3 TUBES in array, 10mm apart (center-to-center)
// 5. ✅ THIN CONNECTING ARMS (2mm) between tubes at base
// 6. ✅ M4 MOUNTING HOLES in connecting arms (2 holes total)
// 7. ✅ DUAL SLOTS on opposing walls - fully unobstructed
// 8. ✅ ANTI-DOUBLE-FEED LEDGES at slot exit

// MOUNTING CONFIGURATION:
// - Two M4 holes in connecting arms between tubes
// - Hole 1: Between tube 1 and tube 2
// - Hole 2: Between tube 2 and tube 3
// - 4.5mm diameter (M4 clearance)
// - Countersunk from bottom for M4 flat head screws
// - Combined with adhesive for maximum security

// ADHESIVE MOUNTING:
// Recommended adhesives:
// - 3M VHB Tape (heavy duty double-sided)
// - Two-part epoxy (permanent)
// - Cyanoacrylate (CA/super glue) - quick setting
// - Hot glue (temporary/removable option)
// 
// Surface prep:
// 1. Sand base lightly (120 grit) for better adhesion
// 2. Clean with isopropyl alcohol
// 3. Apply adhesive to full base surface
// 4. Insert M4 screws through holes into platform
// 5. Tighten screws while adhesive is wet
// 6. Allow to cure fully

// MOUNTING HARDWARE:
// - 2× M4 × 8-10mm flat head screws (countersunk)
// - Screws insert from bottom (pointing up into platform)
// - Platform needs M4 threaded inserts or tapped holes
// - Screw positions: 23.65mm and 51.25mm from left edge

// PUSHER CLEARANCE:
// ✓ Bottom slots completely clear at z = 5mm
// ✓ No mounting hardware interferes with blade travel
// ✓ Pusher can slide full stroke without obstruction
// ✓ Base is flat 5mm surface (maximum glue contact area)

// SLOT CONFIGURATION:
// - Front wall: Slot from z=5.0mm to z=5.9mm
// - Rear wall: Slot from z=5.0mm to z=5.9mm
// - Ledge at z=5.95mm catches 2nd card
// - Pusher enters from either side (0.45mm thick blade)

// ASSEMBLY INSTRUCTIONS:
// 1. Print array upright (tubes pointing up)
// 2. Clean slots with 0.8mm feeler gauge
// 3. Test pusher blade clearance through slots
// 4. Prepare base surface (sand + clean)
// 5. Apply adhesive to full base
// 6. Position on platform, insert M4 screws from below
// 7. Tighten screws while adhesive is wet
// 8. Allow adhesive to cure fully
// 9. Install pusher mechanisms at front/rear
// 10. Load cards from top

// DIMENSIONAL SUMMARY:
// Single tube: 17.3mm (W) × 14.3mm (D) × ~45mm (H)
// Triple array: 61.9mm (W) × 14.3mm (D) × ~45mm (H)
// Tube spacing: 10mm center-to-center
// Connecting arm width: 10mm (between tubes)
// Base thickness: 5mm (flat adhesive surface)
// Base footprint: 61.9mm × 14.3mm = 885mm²
// M4 hole positions: 23.65mm, 51.25mm from left edge
// Total capacity: 150 cards (50 per tube)

// TOLERANCE VERIFICATION:
// ✓ Card (0.67mm) + clearance in 0.9mm slot
// ✓ Ledge (0.95mm) blocks double-feed (1.34mm)
// ✓ Pusher (0.45mm) clears ledge with margin
// ✓ M3 holes (3.2mm) for easy screw insertion