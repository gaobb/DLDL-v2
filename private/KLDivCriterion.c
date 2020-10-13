#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/KLDivCriterion.c"
#else

void THNN_(KLDivCriterion_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *output,
          bool sizeAverage)
{
  THNN_CHECK_NELEMENT(input, target);
  THNN_CHECK_DIM_SIZE(output, 1, 0, 1);
  
  real sum = 0;
  // p(lnp - lnq)
  TH_TENSOR_APPLY2(real, input, real, target,
    real q = *input_data;
    real p = *target_data;
    sum += *target_data > 0 ? *target_data * (log(*target_data) - log(*input_data)) : 0;
  );

  if (sizeAverage)
    sum /= THTensor_(nElement)(input);

  THTensor_(set1d)(output, 0, sum);
}

void THNN_(KLDivCriterion_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *gradInput,
          bool sizeAverage)
{
  THNN_CHECK_NELEMENT(input, target);
  
  real norm = (sizeAverage ? 1./((real)THTensor_(nElement)(input)) : 1.);
  
  THTensor_(resizeAs)(gradInput, input);
  // -p/q
  TH_TENSOR_APPLY3(real, gradInput, real, input, real, target,
    real q = *input_data;
    real p = *target_data;
    *gradInput_data = *target_data > 0 ? norm * (-*target_data / *input_data) : 0;
  );
}

#endif
