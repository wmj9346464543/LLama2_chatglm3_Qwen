from transformers import AutoTokenizer, AutoModel
pretrained_file = r"D:\code\pretrained_models\chatglm3-6b"
tokenizer = AutoTokenizer.from_pretrained(pretrained_file, trust_remote_code=True)
model = AutoModel.from_pretrained(pretrained_file, trust_remote_code=True).half().cuda()#.quantize(4).half()
model = model.eval()
response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=[])
print(response, history)
response, history = model.chat(tokenizer, "晚上睡不着应该吃多少燕麦片", history=history)
print(response)#
