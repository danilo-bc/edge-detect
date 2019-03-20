module sobel3x3det(input [7:0] z1,
					input [7:0] z2,
					input [7:0] z3,
					input [7:0] z4,
					input [7:0] z5,
					input [7:0] z6,
					output reg [11:0] z_out
					);
	reg [7:0] oper1;
	reg [7:0] oper2;
	always @(*) begin
		oper1 = (z1+2*z2+z3);
		oper2 = (z4+2*z5+z6);
		if (oper1>oper2) begin
			z_out = (oper1-oper2);
		end
		else begin
			z_out = (oper2-oper1);
		end

	end

endmodule