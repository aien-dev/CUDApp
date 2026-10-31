# Gate-2 probe: load a cubin through MAX's own DeviceContext (max/v26.7.0
# max.gpu.host.DeviceExternalFunction) and run it with MAX-owned context,
# buffers and stream. Verification tool only; refuses without SPARK_WINDOW.
#
#   max_cubin_probe CUBIN FN MODE
#     MODE run      : load, launch omega_matmul_bf16 ABI (A bf16 16x16, B bf16 16x8,
#                     C f32 16x8), compare to the host fp32 oracle (same loop order as
#                     omega_matmul_cpu_oracle_bf16), exit 0 PASS / 5 numeric
#     MODE loadonly : load + resolve only, exit 0 if accepted
#     MODE reject   : expect load to raise; exit 0 if it raised, 6 if accepted
# Exit 2 usage, 3 no window, 4 load/launch raised (run/loadonly).
from max.gpu.host import DeviceContext
from max.gpu.host.device_context import DeviceExternalFunction
from std.memory import bitcast
from std.os import getenv
from std.sys import argv, exit


def bf16_bits(x: Float32) -> UInt16:
    # x is exactly representable in bf16 for every fixture value (k*0.25, k*0.5)
    return UInt16(bitcast[DType.uint32](x) >> 16)


def bf16_val(h: UInt16) -> Float32:
    return bitcast[DType.float32](UInt32(h) << 16)


def main() raises:
    var args = argv()
    if len(args) != 4:
        print("usage: max_cubin_probe CUBIN FN run|loadonly|reject")
        exit(2)
    var window = getenv("SPARK_WINDOW")
    if window == "":
        print("refused: SPARK_WINDOW is not set; no authorized GPU window")
        exit(3)
    var path = String(args[1])
    var fn_name = String(args[2])
    var mode = String(args[3])
    var cubin: List[UInt8]
    with open(path, "r") as f:
        cubin = f.read_bytes()

    with DeviceContext() as ctx:
        var loaded = True
        var err = String("")
        try:
            var probe = DeviceExternalFunction(
                ctx,
                function_name=fn_name,
                asm=String(StringSlice(unsafe_from_utf8=cubin)),
            )
            _ = probe^
        except e:
            loaded = False
            err = String(e)
        if mode == "reject":
            print("mode=reject file=", path, " fn=", fn_name, " raised=", not loaded, " err=", err, " window=", window, sep="")
            exit(0 if not loaded else 6)
        if not loaded:
            print("FAIL load file=", path, " fn=", fn_name, " err=", err, " window=", window, sep="")
            exit(4)
        if mode == "loadonly":
            print("mode=loadonly file=", path, " fn=", fn_name, " accepted=True window=", window, sep="")
            exit(0)

        comptime M = 16
        comptime K = 16
        comptime N = 8
        var func = DeviceExternalFunction(
            ctx,
            function_name=fn_name,
            asm=String(StringSlice(unsafe_from_utf8=cubin)),
        )
        var a = ctx.enqueue_create_buffer[DType.uint16](M * K)
        var b = ctx.enqueue_create_buffer[DType.uint16](K * N)
        var c = ctx.enqueue_create_buffer[DType.float32](M * N)
        var ha = List[UInt16](capacity=M * K)
        var hb = List[UInt16](capacity=K * N)
        for i in range(M * K):
            ha.append(bf16_bits(Float32((i % 7) - 3) * 0.25))
        for i in range(K * N):
            hb.append(bf16_bits(Float32((i % 5) - 2) * 0.5))
        with a.map_to_host() as ah, b.map_to_host() as bh, c.map_to_host() as ch:
            for i in range(M * K):
                ah[i] = ha[i]
            for i in range(K * N):
                bh[i] = hb[i]
            for i in range(M * N):
                ch[i] = Float32(-12345.0)  # sentinel: an unwritten C fails the check
        ctx.enqueue_function(func, a, b, c, grid_dim=(1,), block_dim=(32,))
        ctx.synchronize()
        var bad = 0
        var worst = Float64(0)
        with c.map_to_host() as ch:
            for i in range(M):
                for j in range(N):
                    var s = Float32(0)
                    for p in range(K):
                        s += bf16_val(ha[i * K + p]) * bf16_val(hb[p * N + j])
                    var d = abs(Float64(ch[i * N + j]) - Float64(s))
                    if d > worst:
                        worst = d
                    if d > 1e-4:
                        bad += 1
        print("mode=run file=", path, " fn=", fn_name, " M=16 K=16 N=8 worst_abs=", worst, " bad=", bad, " window=", window, sep="")
        exit(0 if bad == 0 else 5)
