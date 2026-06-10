# Timing constraints for chiptop
create_clock -period 2.0 -name core_clk [get_ports clk]

# in1 -> RAM data path is the critical one
set_max_delay 1.5 -from [get_ports in1] -to [get_pins u_ram0/D]

# reset is asynchronous: not placement-critical
set_false_path -from [get_ports rst]
