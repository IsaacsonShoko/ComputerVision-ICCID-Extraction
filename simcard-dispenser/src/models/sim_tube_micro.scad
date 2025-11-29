// Micro-SIM gravity tube - parametric
// Units: mm
$fn = 64;

N_CARDS = 50;
card_w = 15.0;    // longer side
card_d = 12.0;    // short side
card_t = 0.76;

clearance_side = 0.2;    // each side
inner_w = card_w + 2 * clearance_side; // 15.4
inner_d = card_d + 2 * clearance_side; // 12.4

wall = 2.5;
slot_h = 0.8;   // slot height
pusher_thk = 0.5;
pusher_width = inner_w - 0.7;
stroke = card_d + 1.0; // push distance (13.0)

stack_h = N_CARDS * card_t + N_CARDS * 0.03 + 5; // small inter-card gap + headroom

// Body
translate([0, 0, 0])
difference() {
  // outer box
  translate([0, 0, 0])
    cube([inner_w + 2 * wall, inner_d + 2 * wall, stack_h + wall + 10], center=false);

  // internal cavity
  translate([wall, wall, 5])
    cube([inner_w, inner_d, stack_h], center=false);

  // bottom slot cutout (on front face)
  translate([wall, -1, 4]) {
    cube([inner_w, 3 + wall, slot_h + 2], center=false);
  }

  // top opening (for loading)
  translate([wall, wall, stack_h + 5])
    cube([inner_w, inner_d, 20], center=false);
}

// Pusher (visual)
translate([wall + (inner_w - pusher_width) / 2, - (pusher_thk + 0.2), 4])
    cube([pusher_width, pusher_thk, slot_h + 1], center=false);

// optional: show a few cards in stack for scale
for (i = [0:4]) {
  translate([wall + 0.1, wall + 0.1, 5 + i * (card_t + 0.03)])
    cube([card_w - 0.2, card_d - 0.2, card_t], center=false);
}