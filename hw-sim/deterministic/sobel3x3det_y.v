module sobel3x3det_y(input [7:0] z1,
					input [7:0] z2,
					input [7:0] z3,
					input [7:0] z7,
					input [7:0] z8,
					input [7:0] z9,
					output [11:0] z_out
					);
	reg [11:0] oper1;
	reg [11:0] oper2;

	abs_diff ad1(.oper1(oper1),
				.oper2(oper2),
				.result(z_out)
				);

	always @(*) begin
		oper1 = (z1+2*z2+z3);
		oper2 = (z7+2*z8+z9);
	end

endmodule