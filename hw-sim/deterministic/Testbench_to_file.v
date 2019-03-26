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

	// Registers and wires for the Sobel module
	reg [7:0] z1;
	reg [7:0] z2;
	reg [7:0] z3;
	reg [7:0] z4;
	reg [7:0] z5;
	reg [7:0] z6;
	reg [7:0] z7;
	reg [7:0] z8;
	reg [7:0] z9;

	wire [11:0] z_out;

	// Auxiliary system signals
	// May be used in the future
	reg clk;
	reg reset;

	// Instantiate device under test (DUT)
	sobel3x3det DUT(.z1(z1),
					 .z2(z2),
					 .z3(z3),
					 .z4(z4),
					 .z5(z5),
					 .z6(z6),
					 .z7(z7),
					 .z8(z8),
					 .z9(z9),
					 .z_out(z_out)
					);

	// Clock may be used later for stochastic
	always begin
		#1 clk <= ~clk;
	end

	initial begin
		// Default inputs for unit tests
		z1 = 8'h01;
		z2 = 8'h00;
		z3 = 8'h00;
		z4 = 8'h00;
		z5 = 8'h00;
		z6 = 8'h00;
		z7 = 8'h00;
		z8 = 8'h00;
		z9 = 8'h00;
		// Initialize 'biggest' auxiliary variable
		biggest = 0;

		#2
		//Load image coded in hexadecimal as memory
		$display("Loading image into memory");
		$readmemh("src.txt",src);

		// 'Crop' area around central pixel z5
		for (i=1;i<src_rows-1;i=i+1) begin
			for(j=1;j<src_cols-1;j=j+1) begin
				// Row above z5
				z1 = src[src_cols*(i-1)+j-1];
				z2 = src[src_cols*(i-1)+j];
				z3 = src[src_cols*(i-1)+j+1];
				// Row of z5
				z4 = src[src_cols*(i)+j-1];
				z5 = src[src_cols*(i)+j];
				z6 = src[src_cols*(i)+j+1];
				// Row below z5
				z7 = src[src_cols*(i+1)+j-1];
				z8 = src[src_cols*(i+1)+j];
				z9 = src[src_cols*(i+1)+j+1];

				#2 // give an instant for z_out to update
				// Update biggest number to normalize later
				if(z_out > biggest) begin
					biggest = z_out;
				end
				edges[src_cols*(i-1)+j-1]=z_out;
			end

		end
		f = $fopen("edges_hw_det.txt","w");
		// Normalize to 8 bits and write to file
		for (i=0;i<src_rows-2;i=i+1) begin
			for(j=0;j<src_cols-2;j=j+1) begin
				edges[src_cols*(i)+j] = (edges[src_cols*(i)+j]*255)/biggest;
				$fwrite(f,"%x ",edges[src_cols*(i)+j][7:0]);
			end
			$fwrite(f,"\n");
		end
		$fclose(f);

		$finish;
	end

endmodule