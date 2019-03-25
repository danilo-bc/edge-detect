module sobel3x3det(input [7:0] z1,
					input [7:0] z2,
					input [7:0] z3,
					input [7:0] z4,
					input [7:0] z5,
					input [7:0] z6,
					input [7:0] z7,
					input [7:0] z8,
					input [7:0] z9,
					output reg [11:0] z_out
					);
	wire [11:0] z_x;
	wire [11:0] z_y;

	always @(*) begin
		z_out = z_x+z_y;
	end

	sobel3x3det_x sobel_x(.z1(z1),
						  .z4(z4),
						  .z7(z7),
						  .z3(z3),
						  .z6(z6),
						  .z9(z9),
						  .z_out(z_x)
						);

	sobel3x3det_y sobel_y(.z1(z1),
						  .z2(z2),
						  .z3(z3),
						  .z7(z7),
						  .z8(z8),
						  .z9(z9),
						  .z_out(z_y)
						);

endmodule