/*
 * This file was generated by the scsynth tool, and is available for use under
 * the MIT license. More information can be found at
 * https://github.com/arminalaghi/scsynth/
 */
module MReSC_Sobel3x3_X_8b_g2( //the stochastic core of an ReSC
	input [0:0] x_1, //independent copies of x_1
	input [0:0] x_2, //independent copies of x_2
	input [0:0] x_3, //independent copies of x_3
	input [0:0] x_4, //independent copies of x_4
	input [0:0] x_5, //independent copies of x_5
	input [0:0] x_6, //independent copies of x_6
	input [63:0] w, //Bernstein coefficients
	output reg z //output bitsream
);

	wire [0:0] sum_1; //sum of x values for mux
	assign sum_1 = x_1[0];

	wire [0:0] sum_2; //sum of x values for mux
	assign sum_2 = x_2[0];

	wire [0:0] sum_3; //sum of x values for mux
	assign sum_3 = x_3[0];

	wire [0:0] sum_4; //sum of x values for mux
	assign sum_4 = x_4[0];

	wire [0:0] sum_5; //sum of x values for mux
	assign sum_5 = x_5[0];

	wire [0:0] sum_6; //sum of x values for mux
	assign sum_6 = x_6[0];

	always @(*) begin
		case(sum_1)
			1'd0: case(sum_2)
				1'd0: case(sum_3)
					1'd0: case(sum_4)
						1'd0: case(sum_5)
							1'd0: case(sum_6)
								1'd0: z = w[0];
								1'd1: z = w[1];
								default: z = 0;
							endcase
							1'd1: case(sum_6)
								1'd0: z = w[2];
								1'd1: z = w[3];
								default: z = 0;
							endcase
							default: z = 0;
						endcase
						1'd1: case(sum_5)
							1'd0: case(sum_6)
								1'd0: z = w[4];
								1'd1: z = w[5];
								default: z = 0;
							endcase
							1'd1: case(sum_6)
								1'd0: z = w[6];
								1'd1: z = w[7];
								default: z = 0;
							endcase
							default: z = 0;
						endcase
						default: z = 0;
					endcase
					1'd1: case(sum_4)
						1'd0: case(sum_5)
							1'd0: case(sum_6)
								1'd0: z = w[8];
								1'd1: z = w[9];
								default: z = 0;
							endcase
							1'd1: case(sum_6)
								1'd0: z = w[10];
								1'd1: z = w[11];
								default: z = 0;
							endcase
							default: z = 0;
						endcase
						1'd1: case(sum_5)
							1'd0: case(sum_6)
								1'd0: z = w[12];
								1'd1: z = w[13];
								default: z = 0;
							endcase
							1'd1: case(sum_6)
								1'd0: z = w[14];
								1'd1: z = w[15];
								default: z = 0;
							endcase
							default: z = 0;
						endcase
						default: z = 0;
					endcase
					default: z = 0;
				endcase
				1'd1: case(sum_3)
					1'd0: case(sum_4)
						1'd0: case(sum_5)
							1'd0: case(sum_6)
								1'd0: z = w[16];
								1'd1: z = w[17];
								default: z = 0;
							endcase
							1'd1: case(sum_6)
								1'd0: z = w[18];
								1'd1: z = w[19];
								default: z = 0;
							endcase
							default: z = 0;
						endcase
						1'd1: case(sum_5)
							1'd0: case(sum_6)
								1'd0: z = w[20];
								1'd1: z = w[21];
								default: z = 0;
							endcase
							1'd1: case(sum_6)
								1'd0: z = w[22];
								1'd1: z = w[23];
								default: z = 0;
							endcase
							default: z = 0;
						endcase
						default: z = 0;
					endcase
					1'd1: case(sum_4)
						1'd0: case(sum_5)
							1'd0: case(sum_6)
								1'd0: z = w[24];
								1'd1: z = w[25];
								default: z = 0;
							endcase
							1'd1: case(sum_6)
								1'd0: z = w[26];
								1'd1: z = w[27];
								default: z = 0;
							endcase
							default: z = 0;
						endcase
						1'd1: case(sum_5)
							1'd0: case(sum_6)
								1'd0: z = w[28];
								1'd1: z = w[29];
								default: z = 0;
							endcase
							1'd1: case(sum_6)
								1'd0: z = w[30];
								1'd1: z = w[31];
								default: z = 0;
							endcase
							default: z = 0;
						endcase
						default: z = 0;
					endcase
					default: z = 0;
				endcase
				default: z = 0;
			endcase
			1'd1: case(sum_2)
				1'd0: case(sum_3)
					1'd0: case(sum_4)
						1'd0: case(sum_5)
							1'd0: case(sum_6)
								1'd0: z = w[32];
								1'd1: z = w[33];
								default: z = 0;
							endcase
							1'd1: case(sum_6)
								1'd0: z = w[34];
								1'd1: z = w[35];
								default: z = 0;
							endcase
							default: z = 0;
						endcase
						1'd1: case(sum_5)
							1'd0: case(sum_6)
								1'd0: z = w[36];
								1'd1: z = w[37];
								default: z = 0;
							endcase
							1'd1: case(sum_6)
								1'd0: z = w[38];
								1'd1: z = w[39];
								default: z = 0;
							endcase
							default: z = 0;
						endcase
						default: z = 0;
					endcase
					1'd1: case(sum_4)
						1'd0: case(sum_5)
							1'd0: case(sum_6)
								1'd0: z = w[40];
								1'd1: z = w[41];
								default: z = 0;
							endcase
							1'd1: case(sum_6)
								1'd0: z = w[42];
								1'd1: z = w[43];
								default: z = 0;
							endcase
							default: z = 0;
						endcase
						1'd1: case(sum_5)
							1'd0: case(sum_6)
								1'd0: z = w[44];
								1'd1: z = w[45];
								default: z = 0;
							endcase
							1'd1: case(sum_6)
								1'd0: z = w[46];
								1'd1: z = w[47];
								default: z = 0;
							endcase
							default: z = 0;
						endcase
						default: z = 0;
					endcase
					default: z = 0;
				endcase
				1'd1: case(sum_3)
					1'd0: case(sum_4)
						1'd0: case(sum_5)
							1'd0: case(sum_6)
								1'd0: z = w[48];
								1'd1: z = w[49];
								default: z = 0;
							endcase
							1'd1: case(sum_6)
								1'd0: z = w[50];
								1'd1: z = w[51];
								default: z = 0;
							endcase
							default: z = 0;
						endcase
						1'd1: case(sum_5)
							1'd0: case(sum_6)
								1'd0: z = w[52];
								1'd1: z = w[53];
								default: z = 0;
							endcase
							1'd1: case(sum_6)
								1'd0: z = w[54];
								1'd1: z = w[55];
								default: z = 0;
							endcase
							default: z = 0;
						endcase
						default: z = 0;
					endcase
					1'd1: case(sum_4)
						1'd0: case(sum_5)
							1'd0: case(sum_6)
								1'd0: z = w[56];
								1'd1: z = w[57];
								default: z = 0;
							endcase
							1'd1: case(sum_6)
								1'd0: z = w[58];
								1'd1: z = w[59];
								default: z = 0;
							endcase
							default: z = 0;
						endcase
						1'd1: case(sum_5)
							1'd0: case(sum_6)
								1'd0: z = w[60];
								1'd1: z = w[61];
								default: z = 0;
							endcase
							1'd1: case(sum_6)
								1'd0: z = w[62];
								1'd1: z = w[63];
								default: z = 0;
							endcase
							default: z = 0;
						endcase
						default: z = 0;
					endcase
					default: z = 0;
				endcase
				default: z = 0;
			endcase
			default: z = 0;
		endcase
	end
endmodule
