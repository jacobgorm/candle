use anyhow::Result;
use candle_core::{Device, Tensor};

fn main() -> Result<()> {
    let device = Device::new_d3d12(0)?;
    println!("D3D12 device created successfully");

    // Test 1: Create tensor and read back
    println!("\n--- Test 1: Create tensor ---");
    let t = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &device)?;
    let vals = t.to_vec1::<f32>()?;
    println!("  tensor: {:?}", vals);
    assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0]);

    // Test 2: Unary neg
    println!("\n--- Test 2: Neg ---");
    let neg = t.neg()?;
    let vals = neg.to_vec1::<f32>()?;
    println!("  neg: {:?}", vals);
    assert_eq!(vals, vec![-1.0, -2.0, -3.0, -4.0]);

    // Test 3: Binary add
    println!("\n--- Test 3: Add ---");
    let a = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &device)?;
    let b = Tensor::new(&[10.0f32, 20.0, 30.0, 40.0], &device)?;
    let c = (&a + &b)?;
    let vals = c.to_vec1::<f32>()?;
    println!("  add: {:?}", vals);
    assert_eq!(vals, vec![11.0, 22.0, 33.0, 44.0]);

    // Test 4: Matmul
    println!("\n--- Test 4: Matmul ---");
    let a = Tensor::new(&[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]], &device)?;
    let b = Tensor::new(&[[7.0f32, 8.0], [9.0, 10.0], [11.0, 12.0]], &device)?;
    let c = a.matmul(&b)?;
    let vals = c.to_vec2::<f32>()?;
    println!("  matmul: {:?}", vals);
    assert_eq!(vals, vec![vec![58.0, 64.0], vec![139.0, 154.0]]);

    // Test 5: Affine (mul + add)
    println!("\n--- Test 5: Affine ---");
    let t = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &device)?;
    let result = t.affine(2.0, 1.0)?;
    let vals = result.to_vec1::<f32>()?;
    println!("  affine(2x+1): {:?}", vals);
    assert_eq!(vals, vec![3.0, 5.0, 7.0, 9.0]);

    // Test 6: Reduce sum
    println!("\n--- Test 6: Sum ---");
    let t = Tensor::new(&[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]], &device)?;
    let sum = t.sum(1)?;
    let vals = sum.to_vec1::<f32>()?;
    println!("  sum(axis=1): {:?}", vals);
    assert!((vals[0] - 6.0).abs() < 1e-4);
    assert!((vals[1] - 15.0).abs() < 1e-4);

    // Test 7: Exp
    println!("\n--- Test 7: Exp ---");
    let t = Tensor::new(&[0.0f32, 1.0], &device)?;
    let e = t.exp()?;
    let vals = e.to_vec1::<f32>()?;
    println!("  exp: {:?}", vals);
    assert!((vals[0] - 1.0).abs() < 1e-5);
    assert!((vals[1] - std::f32::consts::E).abs() < 1e-5);

    // Test 8: Zeros
    println!("\n--- Test 8: Zeros ---");
    let z = Tensor::zeros((2, 3), candle_core::DType::F32, &device)?;
    let vals = z.to_vec2::<f32>()?;
    println!("  zeros: {:?}", vals);
    assert_eq!(vals, vec![vec![0.0; 3]; 2]);

    println!("\n=== All D3D12 tests passed! ===");
    Ok(())
}
