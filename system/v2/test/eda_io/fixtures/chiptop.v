// Structural netlist mirror of floorplan.def (same instances and nets)
module chiptop (clk, rst, in1, out1);
  input clk, rst, in1;
  output out1;

  wire pllout, n0, ramq0, n1, romdo, n2, n3, n4;

  PLL u_pll (.REF(clk), .OUT(pllout));
  RAM16 u_ram0 (.D(n0), .Q(ramq0), .CK(pllout));
  RAM16 u_ram1 (.D(n2), .Q(), .CK(pllout));
  ROM32 u_rom (.A(n2), .DO(romdo));

  INVX1 u0 (.A(in1), .Y(n0));
  NAND2 u1 (.A(ramq0), .B(ramq0), .Y(n1));
  INVX1 u2 (.A(ramq0), .Y());
  NAND2 u3 (.A(n1), .B(n1), .Y());
  INVX1 u4 (.A(romdo), .Y(n2));
  NAND2 u5 (.A(n2), .B(n1), .Y(n4));
  INVX1 u6 (.A(rst), .Y(n3));
  NAND2 u7 (.A(n3), .B(n3), .Y());
  INVX1 u8 (.A(n3), .Y());
  NAND2 u9 (.A(n1), .B(n3), .Y());
  INVX1 u10 (.A(n1), .Y());
  NAND2 u11 (.A(n4), .B(n4), .Y());
  BUFX2 u_buf (.A(n4), .Y(out1));

endmodule
