"""
Attempt to compress physical constants with entropy-reduction algorithms.
"""

def compression_ratio(s):
    import zlib
    compressed = zlib.compress(s.encode())
    return len(compressed) / len(s.encode())

if __name__ == '__main__':
    pi_str = "3.141592653589793238462643383279"
    e_str =  "2.718281828459045235360287471352"
    print("Pi ratio:", compression_ratio(pi_str))
    print("e ratio:", compression_ratio(e_str))