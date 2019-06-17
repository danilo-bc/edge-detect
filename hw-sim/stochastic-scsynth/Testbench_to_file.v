module Testbench_to_file();
	parameter src_rows = 436;
	parameter src_cols = 576;
	//parameter src_rows = 50;
	//parameter src_cols = 50;
	//parameter src_rows = 3;
	//parameter src_cols = 3;
	parameter src_size = src_rows*src_cols;
	parameter edge_size = (src_rows-1)*(src_cols-1);
	reg [7:0] src [0:src_size-1];
	reg [7:0] edges [0:edge_size-1];

	// Variable to save file handle
	integer f;
	// Auxiliary variables for counting in loops
	integer i,j;

	//Registers for Sobel x
	reg [7:0] pixel_1_bin;
	reg [7:0] pixel_2_bin;
	reg [7:0] pixel_3_bin;
	reg [7:0] pixel_4_bin;
	reg [7:0] pixel_6_bin;
	reg [7:0] pixel_7_bin;
	reg [7:0] pixel_8_bin;
	reg [7:0] pixel_9_bin;

	wire [7:0] z_bin; //binary value of output


	reg start;
	wire done;
	reg clk;
	reg reset;

	MReSC_wrapper_Sobel3x3_g2 Sobel(
		.x_1_bin (pixel_1_bin),
		.x_2_bin (pixel_2_bin),
		.x_3_bin (pixel_3_bin),
		.x_4_bin (pixel_4_bin),
		.x_5_bin (pixel_6_bin),
		.x_6_bin (pixel_7_bin),
		.x_7_bin (pixel_8_bin),
		.x_8_bin (pixel_9_bin),
		.start (start),
		.done (done),
		.z_bin (z_bin),
		.clk (clk),
		.reset (reset)
	);

always begin
		#1 clk <= ~clk;
	end

	initial begin
		//Load image coded in hexadecimal as memory
		$dumpfile("sim_output.vcd");
		$dumpvars(0,Testbench_to_file);
		$display("Loading image into memory");
		$readmemh("srcFull.txt",src);
		//$readmemh("square3.txt",src);
		clk = 0;
		reset = 1;
		start = 0;
		#5 reset = 0;
		start = 1;
		#2

		// 'Crop' area around central pixel z5
		for (i=1;i<src_rows-1;i=i+1) begin
			for(j=1;j<src_cols-1;j=j+1) begin

				//3x3 mask for Sobel
				pixel_1_bin <= src[src_cols*(i-1)+j-1];
				pixel_2_bin <= src[src_cols*(i-1)+j];
				pixel_3_bin <= src[src_cols*(i-1)+j+1];
				pixel_4_bin <= src[src_cols*(i)+j-1];
				//Unused central pixel
				pixel_6_bin <= src[src_cols*(i)+j+1];
				pixel_7_bin <= src[src_cols*(i+1)+j-1];
				pixel_8_bin <= src[src_cols*(i+1)+j];
				pixel_9_bin <= src[src_cols*(i+1)+j+1];

				start <= 0;
				#518 ; // give time to calculate


			end
		end

		#10

		f = $fopen("edges_hw_full_scsynthg2.txt","w");
		// Write to file
		for (i=0;i<src_rows-2;i=i+1) begin
			for(j=0;j<src_cols-2;j=j+1) begin
				$fwrite(f,"%x ",edges[src_cols*(i)+j]);
			end
			$fwrite(f,"\n");
		end
		$fclose(f);

		$finish;
	end

	always @(posedge done) begin
		//$display("x: %b, z: %b, expected_z: %b", z_bin, z_bin, expected_z);
		//$fwrite(f,"%b,%b\n", z_bin, expected_z);
		start <= 1;
		edges[src_cols*(i-1)+j-1]<=z_bin;
	end
endmodule
