module mux4(input i1,
			input i2,
			input i3,
			input i4,
			input s1,
			input s2,

			output reg out
			);

always @(*) begin
	case({s1,s2})
		2'b00: out <= i1;
		2'b01: out <= i2;
		2'b10: out <= i3;
		2'b11: out <= i4;
	endcase
end
endmodule