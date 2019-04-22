module sobel3x3det(input [7:0] z1,
					input [7:0] z2,
					input [7:0] z3,
					input [7:0] z4,
					input [7:0] z5,
					input [7:0] z6,
					input [7:0] z7,
					input [7:0] z8,
					input [7:0] z9,
					output reg [7:0] z_out
					);
	wire [11:0] z_x;
	wire [11:0] z_y;

	reg [8:0] aux_x;
	reg [8:0] aux_y;

	always @(*) begin
	/*
	This block uses the same normalization strategy
	of the software counterpart. Saturate values up to
	8-bits (255), copy the rest, then sum 0.5x+0.5y
	z_x and z_y are already in absolute value
	*/
		if(z_x > 12'd255) begin
			aux_x = 9'd255;
		end
		else begin
			aux_x = {1'b0,z_x[7:0]};
		end

		if(z_y > 12'd255) begin
			aux_y = 9'd255;
		end
		else begin
			aux_y = {1'b0,z_y[7:0]};
		end

		// Weighted sum
		z_out = (aux_x+aux_y)>>1;
		
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