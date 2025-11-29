# Design Notes for SIM Card Dispenser Project

## Overview
This document outlines the design considerations and rationale behind the development of the SIM card dispenser project. The dispenser is designed to store and dispense various types of SIM cards, specifically Nano, Micro, and Mini formats.

## Design Considerations

### Dimensions
- **Nano-SIM**: The tube for Nano-SIM cards is designed with internal dimensions of 12.7 mm (width) x 9.2 mm (depth) to accommodate the card's size while allowing for a small clearance to prevent jamming.
- **Micro-SIM**: The Micro-SIM tube has larger internal dimensions of 15.2 mm (width) x 12.2 mm (depth) to fit the Micro-SIM specifications.
- **Mini-SIM**: The Mini-SIM tube is designed with dimensions of 25.2 mm (width) x 15.2 mm (depth) to ensure compatibility with the Mini-SIM format.

### Slot Height
The slot height for all tubes is set at 0.8 mm, which is slightly larger than the thickness of the Nano-SIM (0.67 mm) and accommodates the thickness of Micro and Mini-SIMs (0.76 mm). This design choice ensures that only one card can be dispensed at a time.

### Pusher Mechanism
Each tube features a pusher mechanism designed to push the bottom SIM card out through the slot. The pusher is designed to be thin enough to fit through the slot while being robust enough to apply the necessary force to dispense the card.

### Material Choices
Low-friction materials such as PTFE or smooth plastics are recommended for the tube's interior to minimize friction between the SIM cards and the tube walls. This choice enhances the reliability of the dispensing mechanism.

## Challenges Encountered
- **Tolerance Management**: Ensuring that the dimensions allow for smooth operation without excessive clearance was a key challenge. The design must balance between snug fits and ease of card movement.
- **Material Selection**: Choosing materials that provide the right balance of durability and low friction was critical to the success of the dispenser.

## Conclusion
The design of the SIM card dispenser is a careful balance of dimensions, material choices, and mechanical considerations. The project aims to provide a reliable and efficient solution for storing and dispensing SIM cards of various types. Further testing and prototyping will be essential to refine the design and ensure optimal performance.