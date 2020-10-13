local ExpOut, parent = torch.class('nn.ExpOut', 'nn.Module')

function ExpOut:__init(labelset)
  parent.__init(self)
  self.constant_vector = labelset
end

function ExpOut:updateOutput(input)
    self.output:resize(input:size(1))
    self.output = input * self.constant_vector
  return self.output
end

function ExpOut:updateGradInput(input, gradOutput)
      self.gradInput:resize(input:size())
      local bs, ds = input:size(1), input:size(2)
      self.gradInput:copy(gradOutput:reshape(bs, 1):expand(bs, ds))
      self.gradInput:cmul(torch.reshape(self.constant_vector, 1, ds):expand(bs, ds))
    return self.gradInput
end
