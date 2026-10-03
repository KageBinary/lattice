struct Params { nx:u32, ny:u32, stride:u32, origin:u32, low:f32, high:f32, pad:vec2<u32> };
@group(0) @binding(0) var<uniform> cfg:Params;
@group(0) @binding(1) var<storage,read> field:array<f32>;
@group(0) @binding(2) var<storage,read> colors:array<vec4<f32>>;
@group(0) @binding(3) var output:texture_storage_2d<rgba8unorm,write>;
@compute @workgroup_size(8,8)
fn image(@builtin(global_invocation_id) id:vec3<u32>) {
    if (id.x>=cfg.nx || id.y>=cfg.ny) { return; }
    let value=field[cfg.origin+id.y*cfg.stride+id.x];
    var color=vec4<f32>(1.0,0.0,1.0,1.0);
    if (value>=-3.4028235e38 && value<=3.4028235e38) {
        var fraction=0.5;
        if (cfg.high>cfg.low) { fraction=clamp((value-cfg.low)/(cfg.high-cfg.low),0.0,1.0); }
        color=colors[u32(round(fraction*255.0))];
    }
    textureStore(output,vec2<i32>(i32(id.x),i32(cfg.ny-1u-id.y)),color);
}
