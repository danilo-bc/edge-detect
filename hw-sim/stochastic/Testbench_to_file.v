module Testbench_to_file();
	parameter src_rows = 436;
	parameter src_cols = 576;
	parameter src_size = src_rows*src_cols;
	parameter edge_size = (src_rows-1)*(src_cols-1);
	reg [7:0] src [0:src_size-1];
	reg [11:0] edges [0:edge_size-1];

	// Variable to save file handle
	integer f;
	// Auxiliary variables for counting in loops
	integer i,j;

	// Auxiliary variable to normalize for 8-bit image
	integer biggest;
	
	reg [7:0] x_1_bin; //binary value of input 1
	reg [7:0] x_2_bin; //binary value of input 2
	reg [7:0] x_3_bin; //binary value of input 3
	reg [7:0] x_4_bin; //binary value of input 4
	reg [7:0] x_5_bin; //binary value of input 5
	reg [7:0] x_6_bin; //binary value of input 6
	reg start;
	wire done;
	wire [7:0] z_bin; //binary value of output

	reg clk;
	reg reset;

	MReSC_wrapper_Sobel3x3_X_8b_g2 DUT (
		.x_1_bin (x_1_bin),
		.x_2_bin (x_2_bin),
		.x_3_bin (x_3_bin),
		.x_4_bin (x_4_bin),
		.x_5_bin (x_5_bin),
		.x_6_bin (x_6_bin),
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
		clk = 0;
		biggest = 0;
		reset = 1;
		#5 reset = 0;
		start = 1;
		#10
		//Load image coded in hexadecimal as memory
		$display("Loading image into memory");
		$readmemh("src.txt",src);
		// 'Crop' area around central pixel z5
		for (i=1;i<src_rows-1;i=i+1) begin
			for(j=1;j<src_cols-1;j=j+1) begin
				  // Row above z5
				x_1_bin = src[src_cols*(i-1)+j-1];
				x_2_bin = src[src_cols*(i-1)+j];
				x_3_bin = src[src_cols*(i-1)+j+1];
				  // Row of z5
				/*
				z4 = src[src_cols*(i)+j-1];
				z5 = src[src_cols*(i)+j];
				z6 = src[src_cols*(i)+j+1];
				*/
				  // Row below z5
				x_4_bin = src[src_cols*(i+1)+j-1];
				x_5_bin = src[src_cols*(i+1)+j];
				x_6_bin = src[src_cols*(i+1)+j+1];
				
				start = 0;
				#518 ;// give time to calculate
				
			end

		end
		
		#532

		f = $fopen("edges_hw_stoch_8b.txt","w");
		// Normalize to 8 bits and write to file
		for (i=0;i<src_rows-2;i=i+1) begin
			for(j=0;j<src_cols-2;j=j+1) begin
				//edges[src_cols*(i)+j] = (edges[src_cols*(i)+j]*255)/biggest;
				$fwrite(f,"%x ",edges[src_cols*(i)+j][7:0]);
			end
			$fwrite(f,"\n");
		end
		$fclose(f);

		$finish;
	end

	always @(posedge done) begin
		//$display("x: %b, z: %b, expected_z: %b", x_bin, z_bin, expected_z);
		//$fwrite(f,"%b,%b\n", z_bin, expected_z);
		start = 1;
		if(z_bin > biggest) begin
			biggest = z_bin;
			//$display("Maior: %b", biggest);
		end
		edges[src_cols*(i-1)+j-1]=z_bin;
	end
endmodule
