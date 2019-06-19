module sobel3x3det(input [7:0] z1,
					input [7:0] z2,
					input [7:0] z3,
					input [7:0] z4,
					input [7:0] z6,
					input [7:0] z7,
					input [7:0] z8,
					input [7:0] z9,
					input clk,
					input reset,
					output reg [7:0] z_out
					);
	wire [9:0]abs1;
	wire [9:0]abs2;
	reg  [9:0] neg1;
	reg  [9:0] neg2;
	reg  [9:0] pos1;
	reg  [9:0] pos2;

	always@(*) begin
		neg1 = (z1+2*z4+z7);
		pos1 = (z3+2*z6+z9);
		neg2 = (z1+2*z2+z3);
		pos2 = (z7+2*z8+z9);
	end

	
	abs_diff A1(	.oper1(neg1),
					.oper2(pos1),
					.result(abs1)
					);
	abs_diff A2(	.oper1(neg2),
					.oper2(pos2),
					.result(abs2)
					);
					
	always @(posedge clk or posedge reset) begin
		if(reset)begin
			z_out <= 8'h00;
		end
		
		else begin
			z_out<= ((abs1+abs2)/8);
		end
	end


endmodule