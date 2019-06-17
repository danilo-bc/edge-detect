module Testbench_to_file();
	integer i;
	integer f;

	//reg start;
	wire done;

	reg init;
	reg running;
	reg clk;
	reg reset;

	wire [3:0] randx_1_0;
	LFSR_4_bit_added_zero_Sobel3x3_X_4b_g2 rand_gen_x_1_0 (
		.seed (4'd0),
		.data (randx_1_0),
		.enable (running),
		.restart (init),
		.clk (clk),
		.reset (reset)
	);

	wire [7:0] randx_2_0;
	LFSR_8_bit_added_zero_Sobel3x3_X_8b_g2 rand_gen_x_2_0 (
		.seed (8'd0),
		.data (randx_2_0),
		.enable (running),
		.restart (init),
		.clk (clk),
		.reset (reset)
	);


	// Simulation:
	always begin
		#1 clk <= ~clk;
	end

	initial begin
		f = $fopen("lfsr4b8bOracle.data","w");
		clk = 0;
		reset = 1;
		init = 0;
		running = 0;
		#5 reset = 0;
		running = 1;
		$display("LFSR 1");
		$display("%b",randx_1_0);
		$fwrite(f,"%b\n",randx_1_0);
		for(i=0;i<15;i=i+1) begin
			#2
			$display("%b",randx_1_0);
			$fwrite(f,"%b\n",randx_1_0);
		end

		#4 reset = 1;
		#1 reset = 0;
		$display("LFSR 2");
		$display("%b",randx_2_0,);
		$fwrite(f,"%b\n",randx_2_0);
		for(i=0;i<15;i=i+1) begin
			#2
			$display("%b",randx_2_0);
			$fwrite(f,"%b\n",randx_2_0);
		end
		#10

		$fclose(f);

		$finish;
	end

endmodule
