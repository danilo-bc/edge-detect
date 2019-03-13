module Testbench_to_file();
	parameter src_rows = 436;
	parameter src_cols = 576;
	parameter src_size = src_rows*src_cols;
	reg [7:0] src [0:src_size-1];
	//reg [7:0] edges [0:217];

	// Variable to save file handle
	integer f;
	// Auxiliary variables for counting in loops
	integer i,j;

	wire [7:0] z [0:5];
	wire [7:0] z_out; //binary value of output

	reg clk;
	reg reset;

	sobel3x3det det1(.z1(z[0]),
					 .z2(z[1]),
					 .z3(z[2]),
					 .z4(z[3]),
					 .z5(z[4]),
					 .z6(z[5]),
					 .z_out(z_out)
					);

	always begin
		#1 clk <= ~clk;
	end

	initial begin
		//Load image coded in hexadecimal as memory
		$display("z_out = %b",z_out);
		$display("Loading image into memory");
		$readmemh("src.txt",src);
		f = $fopen("src_after_load.txt","w");
		for (i=0;i<src_rows;i=i+1)
		begin
			for(j=0;j<src_cols-1;j=j+1)
			begin
				$fwrite(f,"%x ",src[src_cols*i+j]);
			end
			$fwrite(f,"%x\n",src[src_cols*i+src_cols-1]);
		end
		$fclose(f);

		//$display("Loading reference image into memory");
		//$readmemh("edges.txt",edges);
//TODO: automatically compare edges.txt and output of test up until now

		//$stop;
		$finish;
	end

endmodule