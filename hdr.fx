#include "common.fxh"
#include "postprocessing.fxh"

static const float3 LUMINANCE_VECTOR  = float3(0.2125f, 0.7154f, 0.0721f);
static const float3 BLUE_SHIFT_VECTOR = float3(1.05f, 0.97f, 1.27f); 
static const float  MIDDLE_GRAY = 0.07f;
static const float  LUM_WHITE = 1.0f;
static const float  BRIGHT_THRESHOLD = 0.6f;

Texture2D luminanceTex;
Texture2D bloomTex;

cbuffer HDRConstants
{
	
}

static const float blurWeights[15] = { 0.15724300, 0.13310331, 0.10082112, 0.068337522, 0.041448805, 0.022496235, 0.010925785,
 0.13298075, //Center texel
    0.15724300, 0.13310331, 0.10082112, 0.068337522, 0.041448805, 0.022496235, 0.010925785};

/**
Convert source texture to luminance by taking 3x3 averages
*/
PS_OUTPUT_SIMPLE_R32 PS_sourceToLum(PS_INPUT_SIMPLE input)
{
	PS_OUTPUT_SIMPLE_R32 output = (PS_OUTPUT_SIMPLE_R32)0;
				
	float2 coords = input.pos.xy/viewPort.xy;
	
	float4 vColor = 0.0f;
    float  fAvg = 0.0f;
    
    [unroll] for( int y = -1; y < 1; y++ )
    {
        [unroll] for( int x = -1; x < 1; x++ )
        {
            // Compute the sum of color values
            vColor = inputTexture.Sample(samPointClamp,coords,int2(x,y)); 
            fAvg += dot( vColor.rgb, LUMINANCE_VECTOR );
        }
    }
    
    fAvg /= 4;

    output.color = fAvg;
	
	return output;
}

/**
Downscale luminance texture 3x3 times
*/
PS_OUTPUT_SIMPLE_R32 PS_downscaleLum(PS_INPUT_SIMPLE input)
{
	PS_OUTPUT_SIMPLE_R32 output = (PS_OUTPUT_SIMPLE_R32)0;
				
	float2 coords = input.pos.xy/viewPort.xy;
	
    [unroll] for( int y = -1; y < 2; y++ )
    {
        [unroll] for( int x = -1; x < 2; x++ )
        {
            // Compute the sum of color values
             output.color += inputTexture.Sample(samPointClamp,coords,int2(x,y)).r; 
        }
    }
    
    output.color /= 9;
	return output;
}

/**
Luminance adaptation
*/
PS_OUTPUT_SIMPLE_R32 PS_calculateAdaptedLum(PS_INPUT_SIMPLE input)
{
	PS_OUTPUT_SIMPLE_R32 output = (PS_OUTPUT_SIMPLE_R32)0;
	float fAdaptedLum = inputTexture.Sample(samPointClamp, float2(0, 0) ).r;
	float fCurrentLum = luminanceTex.Sample(samPointClamp, float2(0, 0) ).r;
	
	// The user's adapted luminance level is simulated by closing the gap between
	// adapted luminance and current luminance by .5% every frame, based on a
	// 30 fps rate. This is not an accurate model of human adaptation, which can
	// take longer than half an hour.
	float fNewAdaptation = fAdaptedLum + (fCurrentLum - fAdaptedLum) * ( 1 - pow( 0.995f, 30 * elapsedTime ) );
	output.color = fNewAdaptation;
	
	return output;
}

/**
Bright pass
*/
PS_OUTPUT_SIMPLE PS_brightPass(PS_INPUT_SIMPLE input)
{
	PS_OUTPUT_SIMPLE output = (PS_OUTPUT_SIMPLE)0;
				
	float2 coords = input.pos.xy/viewPort.xy;
	
    float3 vColor = 0.0f;    
    float  fLum = luminanceTex.Sample(samPointClamp, float2(0, 0) ).r;
       
    [unroll] for( int y = -1; y < 2; y++ ) 
    {
        [unroll] for( int x = -1; x < 2; x++ )
        {
            // Compute the sum of color values
            float4 vSample = inputTexture.Sample( samPointClamp, coords, int2(x,y) );                       
            vColor += vSample.rgb;
        }
    }
    
    // Divide the sum to complete the average
    vColor /= 9;
 
    // Bright pass and tone mapping
    vColor = max( 0.0f, vColor - BRIGHT_THRESHOLD );
    vColor *= MIDDLE_GRAY / (fLum + 0.001f);
    vColor *= (1.0f + vColor/LUM_WHITE);
    vColor /= (1.0f + vColor);
    
    output.color = float4(vColor, 1.0f);
	return output;
}

/**
Gaussian blur
*/
float4 gaussianBlur(Texture2D tex, float2 coords, bool horiz)
{
	float4 result=0;
	float4 vColor;

	[unroll] for( int iSample = 0; iSample < 15; iSample++ )
	{
		// Sample from adjacent points
		int2 offset=int2(0,iSample-7);
		if(horiz)
			offset.xy=offset.yx;
		vColor = tex.Sample( samPointClamp, coords, offset);
        
		result += vColor*blurWeights[iSample];
	}
	return result;
}

PS_OUTPUT_SIMPLE PS_blur_horiz(PS_INPUT_SIMPLE input)
{
	PS_OUTPUT_SIMPLE output = (PS_OUTPUT_SIMPLE)0;
	float2 coords = input.pos.xy/viewPort.xy;
	output.color = gaussianBlur(inputTexture,coords,true);
	return output;
}

PS_OUTPUT_SIMPLE PS_blur_vert(PS_INPUT_SIMPLE input)
{
	PS_OUTPUT_SIMPLE output = (PS_OUTPUT_SIMPLE)0;
	float2 coords = input.pos.xy/viewPort.xy;
	output.color = gaussianBlur(inputTexture,coords,false);
	return output;
}


float3 Uncharted2Tonemap(float3 x)
{
	float U2_A = 0.18f;	// Shoulder strength, 0.22
	float U2_B = 0.30f; // Linear strength
	float U2_C = 0.10f;	// Linear angle
	float U2_D = 0.20f; // Toe strength
	float U2_E = 0.01f; // Toe numerator
	float U2_F = 0.22f; // Toe denominator
	
	return ((x*(U2_A*x+U2_C*U2_B)+U2_D*U2_E)/(x*(U2_A*x+U2_B)+U2_D*U2_F))-U2_E/U2_F;
}

PS_OUTPUT_SIMPLE PS_final(PS_INPUT_SIMPLE input)
{

	

	PS_OUTPUT_SIMPLE output = (PS_OUTPUT_SIMPLE)0;
	float2 coords = input.pos.xy/viewPort.xy;
	output.color = inputTexture.Sample( samPointClamp, coords);
    float vLum = luminanceTex.Sample( samPointClamp, float2(0,0) ).r;
    float4 vBloom = bloomTex.Sample( samClamp, coords );
    
	// Define a linear blending from -1.5 to 2.6 (log scale) which
	// determines the lerp amount for blue shift
    float fBlueShiftCoefficient = 1.0f - (vLum + 1.5)/1.1;
    fBlueShiftCoefficient = saturate(fBlueShiftCoefficient);

	// Lerp between current color and blue, desaturated copy
    float3 vRodColor = dot( (float3)output.color, LUMINANCE_VECTOR ) * BLUE_SHIFT_VECTOR;
    output.color.rgb = lerp( (float3)output.color, vRodColor, fBlueShiftCoefficient );

    // Tone mapping
    // output.color.rgb *= MIDDLE_GRAY / (vLum + 0.001f);
    // output.color.rgb *= (1.0f + output.color.rgb/LUM_WHITE);
    // output.color.rgb /= (1.0f + output.color.rgb);
	
	// U2 tonemap
	output.color.rgb = pow(output.color.rgb,2.2f);	// Pre-gamma
		
    output.color.rgb *= MIDDLE_GRAY / (vLum + 0.001f); // Exposure (HDR)
	
	float ExposureBias = 2.0f;
	output.color.rgb = Uncharted2Tonemap(ExposureBias*output.color.rgb);
	
	float3 whiteScale = 1.0f/Uncharted2Tonemap(6.0f);
	output.color.rgb *= whiteScale;
	
	output.color.rgb = pow(output.color.rgb,1.0f/2.2f);	// Post-gamma
	
    output.color.rgb += 0.4f * vBloom.rgb;
    output.color.a = 1.0f;
    
	return output;
}

/////////////////////////////////////////////////////////////
//TECHNIQUES
/////////////////////////////////////////////////////////////
technique10 sourceToLum
{
	pass P0
	{
		SetVertexShader( CompileShader( vs_4_0, VS_identity() ) );	
		SetGeometryShader( 0 );
		SetPixelShader( CompileShader( ps_4_0, PS_sourceToLum() ) );
		SetRasterizerState(rstate_NoMSAA);    	
	}
}

technique10 downscaleLum
{
	pass P0
	{
		SetVertexShader( CompileShader( vs_4_0, VS_identity() ) );	
		SetGeometryShader( 0 );
		SetPixelShader( CompileShader( ps_4_0, PS_downscaleLum() ) );
		SetRasterizerState(rstate_NoMSAA);    	
	}
}

technique10 brightPass
{
	pass P0
	{
		SetVertexShader( CompileShader( vs_4_0, VS_identity() ) );	
		SetGeometryShader( 0 );
		SetPixelShader( CompileShader( ps_4_0, PS_brightPass() ) );
		SetRasterizerState(rstate_NoMSAA);    	
	}
}

technique10 blur
{
	pass P0
	{
		SetVertexShader( CompileShader( vs_4_0, VS_identity() ) );	
		SetGeometryShader( 0 );
		SetPixelShader( CompileShader( ps_4_0, PS_blur_horiz() ) );
		SetRasterizerState(rstate_NoMSAA);    	
	}
	pass P1
	{
		SetVertexShader( CompileShader( vs_4_0, VS_identity() ) );	
		SetGeometryShader( 0 );
		SetPixelShader( CompileShader( ps_4_0, PS_blur_vert() ) );
		SetRasterizerState(rstate_NoMSAA);    	
	}
}

technique10 finalPass
{
	pass P0
	{
		SetVertexShader( CompileShader( vs_4_0, VS_identity() ) );	
		SetGeometryShader( 0 );
		SetPixelShader( CompileShader( ps_4_0, PS_final() ) );
		SetRasterizerState(rstate_NoMSAA);    	
	}
}

technique10 adaptiveLum
{
	pass P0
	{
		SetVertexShader( CompileShader( vs_4_0, VS_identity() ) );	
		SetGeometryShader( 0 );
		SetPixelShader( CompileShader( ps_4_0, PS_calculateAdaptedLum() ) );
		SetRasterizerState(rstate_NoMSAA);    	
	}
}

