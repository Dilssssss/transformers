<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Pipeline

[`Pipeline`], Hugging Face [Hub](https://hf.co/models) üzerindeki herhangi bir modelle çeşitli makine öğrenimi görevleri için kullanılabilecek basit ama güçlü bir çıkarım (inference) API'sidir.

[`Pipeline`]'ı, otomatik konuşma tanıma (ASR) için toplantı notlarını transkribe ederken zaman damgaları eklemek gibi göreve özel parametrelerle özelleştirebilirsiniz. [`Pipeline`], çıkarımı hızlandırmak ve bellekten tasarruf etmek için GPU'ları, Apple Silicon'u ve yarı duyarlıklı (half-precision) ağırlıkları destekler.

<Youtube id=tiZFewofSLM/>

Transformers'ın iki tür pipeline sınıfı vardır: genel bir [`Pipeline`] ve [`TextGenerationPipeline`] veya [`VisualQuestionAnsweringPipeline`] gibi birçok göreve özel pipeline. Bu göreve özel pipeline'ları, [`Pipeline`] içindeki `task` parametresine görev tanımlayıcısını yazarak yükleyebilirsiniz. Her pipeline için görev tanımlayıcısını API dokümantasyonunda bulabilirsiniz.

Her görev, varsayılan olarak önceden eğitilmiş bir model ve ön işlemci kullanacak şekilde yapılandırılmıştır, ancak farklı bir model kullanmak isterseniz `model` parametresiyle bu ayarı geçersiz kılabilirsiniz.

Örneğin, [Gemma 2](./model_doc/gemma2) modeliyle [`TextGenerationPipeline`]'ı kullanmak için `task="text-generation"` ve `model="google/gemma-2-2b"` parametrelerini ayarlayın.

```py
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="google/gemma-2-2b")
pipeline("gerçekten iyi bir kek pişirmenin sırrı ")
[{'generated_text': 'gerçekten iyi bir kek pişirmenin sırrı 1. doğru malzemeler 2.'}]
```

Birden fazla girdiniz olduğunda, bunları bir liste olarak geçirin.

```py
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="google/gemma-2-2b", device="cuda")
pipeline(["gerçekten iyi bir kek pişirmenin sırrı ", "bir baget "])
[[{'generated_text': 'gerçekten iyi bir kek pişirmenin sırrı 1. doğru malzemeler 2. '}],
 [{'generated_text': 'bir baget %100 ekmektir.\n\nbir baget %100'}]]
```

Bu rehber, size [`Pipeline`]'ı tanıtacak, özelliklerini gösterecek ve çeşitli parametrelerini nasıl yapılandıracağınızı anlatacak.

## Görevler

[`Pipeline`], farklı modalitelerdeki birçok makine öğrenimi göreviyle uyumludur. Pipeline'a uygun bir girdi verin, gerisini o halledecektir.

İşte [`Pipeline`]'ı farklı görevler ve modaliteler için kullanmanın bazı örnekleri:

<hfoptions id="tasks">
<hfoption id="summarization">

```py
from transformers import pipeline

pipeline = pipeline(task="summarization", model="google/pegasus-billsum")
pipeline("Bu bölüm önceden bu başlığın 44. bölümü olarak belirlenmişti. İlk çıkarıldığı haliyle, bu bölüm 'Bu kanun, 30 Haziran 1864'te onaylanan 'Kaliforniya Eyaletine Yosemite Vadisi ve Mariposa Büyük-Ağaç Korusu'nu kapsayan arazinin bağışını yetkilendiren kanun uyarınca Kaliforniya Eyaletine yapılan arazi bağışını hiçbir şekilde etkilemez' şeklinde iki ek hüküm içeriyordu. İlk alıntılanan hüküm, Kanun'un atıfta bulunduğu arazi Kaliforniya eyaletine devredildikten sonra ABD'ye geri verildiği için Kanun'dan çıkarıldı. 11 Haziran 1906 tarihli 27 No'lu Karar, bu devri kabul etti.")
[{'summary_text': 'İçişleri Bakanına, Kaliforniya'daki Yosemite ve Mariposa Ulusal Ormanları sınırları içinde yer alan belirli arazilerdeki ABD'nin tüm hak, mülkiyet ve menfaatlerini Kaliforniya Eyaletine devretmesini talep eder.'}]
```

</hfoption>
<hfoption id="automatic speech recognition">

```py
from transformers import pipeline

pipeline = pipeline(task="automatic-speech-recognition", model="openai/whisper-large-v3")
pipeline("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
{'text': ' Bir hayalim var, bir gün bu ulus ayağa kalkacak ve inancının gerçek anlamını yaşayacak.'}
```

</hfoption>
<hfoption id="image classification">

```py
from transformers import pipeline

pipeline = pipeline(task="image-classification", model="google/vit-base-patch16-224")
pipeline(images="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
[{'label': 'vaşak', 'score': 0.43350091576576233},
 {'label': 'puma, dağ aslanı, panter, Felis concolor',
  'score': 0.034796204417943954},
 {'label': 'kar leoparı, Panthera uncia',
  'score': 0.03240183740854263},
 {'label': 'Mısır kedisi', 'score': 0.02394474856555462},
 {'label': 'kaplan kedisi', 'score': 0.02288915030658245}]
```

</hfoption>
<hfoption id="visual question answering">

```py
from transformers import pipeline

pipeline = pipeline(task="visual-question-answering", model="Salesforce/blip-vqa-base")
pipeline(
    image="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-few-shot.jpg",
    question="Görselde ne var?",
)
[{'answer': 'özgürlük heykeli'}]
```

</hfoption>
</hfoptions>

## Parametreler

[`Pipeline`] en temelde yalnızca bir görev tanımlayıcısı, model ve uygun girdi gerektirir. Ancak pipeline'ı yapılandırmak için, göreve özel parametrelerden performans optimizasyonuna kadar pek çok parametre mevcuttur.

Bu bölüm size bazı önemli parametreleri tanıtacak.

### Donanım (Device)

[`Pipeline`] GPU'lar, CPU'lar, Apple Silicon ve daha fazlası dahil olmak üzere birçok donanım türüyle uyumludur. Donanım türünü `device` parametresiyle yapılandırabilirsiniz. Varsayılan olarak [`Pipeline`], `device=-1` ile belirtilen CPU üzerinde çalışır.

<hfoptions id="device">
<hfoption id="GPU">

[`Pipeline`]'ı bir GPU üzerinde çalıştırmak için `device` parametresini ilgili CUDA cihaz ID'si olarak ayarlayın. Örneğin, `device=0` birinci GPU'da çalıştırır.

```py
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="google/gemma-2-2b", device=0)
pipeline("gerçekten iyi bir kek pişirmenin sırrı ")
```

Dağıtılmış eğitim için bir kütüphane olan [Accelerate](https://hf.co/docs/accelerate/index)'i kullanarak model ağırlıklarının uygun donanımda nasıl yükleneceğini ve saklanacağını otomatik olarak seçmesini de sağlayabilirsiniz. Bu özellikle birden fazla donanımınız varsa oldukça kullanışlıdır. Accelerate, model ağırlıklarını önce en hızlı donanıma yükler, ardından gerektiğinde bu ağırlıkları diğer donanımlara (CPU, sabit disk) taşır. Otomatik seçim yapması için `device_map="auto"` olarak ayarlayın.

[!İPUCU]
> [Accelerate](https://hf.co/docs/accelerate/basic_tutorials/install) kütüphanesinin kurulu olduğundan emin olun.
>
> ```py
> !pip install -U accelerate
> ```

```py
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="google/gemma-2-2b", device_map="auto")
pipeline("gerçekten iyi bir kek pişirmenin sırrı ")
```

</hfoption>
<hfoption id="Apple silicon">

[Pipeline]'ı Apple silicon üzerinde çalıştırmak için device="mps" olarak ayarlayın.
```py
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="google/gemma-2-2b", device="mps")
pipeline("gerçekten iyi bir kek pişirmenin sırrı ")
```

</hfoption>
</hfoptions>

### Toplu çıkarım (Batch inference)

[`Pipeline`], `batch_size` parametresiyle girdilerin toplu halde işlenmesini de sağlayabilir. Toplu çıkarım özellikle GPU üzerinde hız artışı sağlayabilir, ancak bu garanti edilmez. Donanım, veri ve modelin kendisi gibi diğer faktörler, toplu çıkarımın hızı artırıp artırmayacağını etkileyebilir. Bu nedenle toplu çıkarım varsayılan olarak devre dışıdır.

Aşağıdaki örnekte, 4 girdi olduğunda ve `batch_size` 2 olarak ayarlandığında, [`Pipeline`] her seferinde 2 girdilik bir grubu modele iletir.

```py
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="google/gemma-2-2b", device="cuda", batch_size=2)
pipeline(["gerçekten iyi bir kek pişirmenin sırrı", "bir baget", "paris dünyanın", "sosisli sandviçler"])
[[{'generated_text': 'gerçekten iyi bir kek pişirmenin sırrı iyi bir kek karışımı kullanmaktır.\n\nBen'}],
 [{'generated_text': 'bir baget'}],
 [{'generated_text': 'paris dünyanın en güzel şehridir.\n\nParis\'e 3'}],
 [{'generated_text': 'sosisli sandviçler amerikan mutfağının temelidir. Protein açısından zengindir ve'}]]
```

Toplu çıkarımın bir diğer iyi kullanım alanı da [`Pipeline`]'da veri akışı (streaming) işlemleridir.

```py
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
import datasets

# KeyDataset, veri kümesi tarafından döndürülen sözlük içindeki öğeleri döndüren bir yardımcı araçtır
dataset = datasets.load_dataset("imdb", name="plain_text", split="unsupervised")
pipeline = pipeline(task="text-classification", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", device="cuda")
for out in pipeline(KeyDataset(dataset, "text"), batch_size=8, truncation="only_first"):
    print(out)
```

Toplu çıkarımın performansı artırıp artırmayacağını değerlendirirken aşağıdaki genel kuralları aklınızda bulundurun:

1. Kesin sonucu öğrenmenin tek yolu, modeliniz, verileriniz ve donanımınız üzerinde performansı ölçmektir.
2. Gecikmeye duyarlı senaryolarda (canlı çıkarım ürünleri gibi) toplu çıkarım kullanmayın.
3. CPU kullanıyorsanız toplu çıkarım yapmayın.
4. Verilerinizin `sequence_length` değerini bilmiyorsanız toplu çıkarım yapmayın. Performansı ölçün, `sequence_length` değerini aşamalı olarak artırın ve bellek taşması (OOM) kontrolleri ekleyerek hatalardan kurtulun.
5. `sequence_length` değeriniz sabitse toplu çıkarım yapın ve OOM hatası alana kadar bu değeri artırmaya devam edin. GPU ne kadar büyükse, toplu çıkarım o kadar faydalı olacaktır.
6. Toplu çıkarım yapmaya karar verirseniz, OOM hatalarını yönetebileceğinizden emin olun.

### Göreve özel parametreler

[`Pipeline`], her bir görev pipeline'ı tarafından desteklenen tüm parametreleri kabul eder. Hangi parametrelerin mevcut olduğunu görmek için ilgili görev pipeline'ının dokümantasyonunu inceleyin. Eğer ihtiyacınız olan bir parametreyi bulamazsanız, GitHub üzerinden bir [istek](https://github.com/huggingface/transformers/issues/new?assignees=&labels=feature&template=feature-request.yml) oluşturmaktan çekinmeyin!

Aşağıdaki örnekler, mevcut olan bazı göreve özel parametreleri göstermektedir.

<hfoptions id="task-specific-parameters">
<hfoption id="automatic speech recognition">

Her kelimenin ne zaman söylendiğini döndürmesi için [`Pipeline`]'a `return_timestamps="word"` parametresini iletin.

```py
from transformers import pipeline

pipeline = pipeline(task="automatic-speech-recognition", model="openai/whisper-large-v3")
pipeline(audio="https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac", return_timestamp="word")
{'text': ' Bir hayalim var, bu ulus bir gün ayağa kalkacak ve inancının gerçek anlamını yaşayacak.',
 'chunks': [{'text': ' Bir', 'timestamp': (0.0, 1.1)},
  {'text': ' hayalim', 'timestamp': (1.1, 1.44)},
  {'text': ' var', 'timestamp': (1.44, 1.62)},
  {'text': ' bu', 'timestamp': (1.62, 1.92)},
  {'text': ' ulus', 'timestamp': (1.92, 3.7)},
  {'text': ' bir', 'timestamp': (3.7, 3.88)},
  {'text': ' gün', 'timestamp': (3.88, 4.24)},
  {'text': ' ayağa', 'timestamp': (4.24, 5.82)},
  {'text': ' kalkacak', 'timestamp': (5.82, 6.78)},
  {'text': ' ve', 'timestamp': (6.78, 7.36)},
  {'text': ' inancının', 'timestamp': (7.36, 7.88)},
  {'text': ' gerçek', 'timestamp': (7.88, 8.46)},
  {'text': ' anlamını', 'timestamp': (8.46, 9.2)},
  {'text': ' yaşayacak.', 'timestamp': (9.2, 10.34)}]}
```

</hfoption>
<hfoption id="text generation">

Sadece oluşturulan metni (tüm metni değil - yani prompt ve oluşturulan metin yerine) döndürmesi için [`Pipeline`]'a `return_full_text=False` parametresini iletin.

[`~TextGenerationPipeline.__call__`] ayrıca [`~GenerationMixin.generate`] metodundan ek anahtar kelime argümanlarını destekler. Birden fazla oluşturulmuş dizi döndürmek için `num_return_sequences` değerini 1'den büyük bir değer ayarlayın.

```py
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="openai-community/gpt2")
pipeline("the secret to baking a good cake is", num_return_sequences=4, return_full_text=False)
[{'generated_text': ' bunu ellerimle yapmanın ne kadar kolay olduğu. Çıldırmamalısınız, yoksa kek düşecektir.'},
 {'generated_text': ' keki pişirmeden önce hazırlamak. Püf noktası, doğru türde bir krema bulmak ve bu kremanın harika bir kaplama yapmasıdır.\n\nİyi bir krema kaplama için size temel bilgileri veriyoruz'},
 {'generated_text': " yeterince suda bekletmeyi unutmamak ve duvara yapışması konusunda endişelenmemek. Bu arada, kekin üst kısmını çıkarıp bir kağıt havluyla kurumaya bırakabilirsiniz.\n"},
 {'generated_text': ' fırını kapatmak ve 30 dakika bekletmek için en iyi zaman. 30 dakika sonra karıştırın ve tamamen nemli olana kadar bir tavada kek pişirin.\n\nKeki yaklaşık 12'}]
```

</hfoption>
</hfoptions>

## Parçalı Toplu İşleme (Chunk Batching)

Bazı durumlarda verileri parçalar halinde işlemeniz gerekebilir:

- Bazı veri türlerinde (örneğin çok uzun bir ses dosyası), tek bir girdi işlenmeden önce birden fazla parçaya bölünmelidir
- Sıfırdan sınıflandırma (zero-shot) veya soru-cevap gibi bazı görevlerde, tek bir girdi birden fazla ileri geçiş gerektirebilir ve bu da `batch_size` parametresiyle sorunlara yol açabilir

[ChunkPipeline](https://github.com/huggingface/transformers/blob/99e0ab6ed888136ea4877c6d8ab03690a1478363/src/transformers/pipelines/base.py#L1387) sınıfı bu kullanım senaryolarını ele almak üzere tasarlanmıştır. Her iki pipeline sınıfı da aynı şekilde kullanılır, ancak [ChunkPipeline](https://github.com/huggingface/transformers/blob/99e0ab6ed888136ea4877c6d8ab03690a1478363/src/transformers/pipelines/base.py#L1387) otomatik olarak toplu işlemeyi yönetebildiği için, girdilerinizin tetiklediği ileri geçiş sayısı konusunda endişelenmenize gerek yoktur. Bunun yerine, girdilerden bağımsız olarak `batch_size` değerini optimize edebilirsiniz.

Aşağıdaki örnek, bunun [`Pipeline`]'dan nasıl farklılaştığını göstermektedir.

```py
# ChunkPipeline
all_model_outputs = []
for preprocessed in pipeline.preprocess(inputs):
    model_outputs = pipeline.model_forward(preprocessed)
    all_model_outputs.append(model_outputs)
outputs =pipeline.postprocess(all_model_outputs)

# Pipeline
preprocessed = pipeline.preprocess(inputs)
model_outputs = pipeline.forward(preprocessed)
outputs = pipeline.postprocess(model_outputs)
```

## Büyük veri kümeleri

Büyük veri kümeleriyle çıkarım yaparken, doğrudan veri kümesi üzerinde döngü oluşturabilirsiniz. Bu yaklaşım:

- Tüm veri kümesi için belleği hemen tahsis etmeyi önler
- Toplu işlem oluşturma endişesini ortadan kaldırır

Performansı artırıp artırmadığını görmek için [`batch_size` parametresiyle Toplu çıkarım](#batch-inference) deneyebilirsiniz.

```py
from transformers.pipelines.pt_utils import KeyDataset
from transformers import pipeline
from datasets import load_dataset

dataset = datasets.load_dataset("imdb", name="plain_text", split="unsupervised")
pipeline = pipeline(task="text-classification", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", device="cuda")
for out in pipeline(KeyDataset(dataset, "text"), batch_size=8, truncation="only_first"):
    print(out)
```

[`Pipeline`] ile büyük veri kümelerinde çıkarım yapmanın diğer yöntemleri arasında bir **iterator** (yineleyici) veya **generator** (üreteç) kullanımı yer alır.

```py
def data():
    for i in range(1000):
        yield f"My example {i}"

pipeline = pipeline(model="openai-community/gpt2", device=0)
generated_characters = 0
for out in pipeline(data()):
    generated_characters += len(out[0]["generated_text"])
```

## Büyük Ölçekli Modeller

[Accelerate](https://hf.co/docs/accelerate/index) sayesinde [`Pipeline`] ile büyük modelleri çalıştırırken çeşitli iyileştirmeler yapabilirsiniz. Başlamadan önce Accelerate'i kurduğunuzdan emin olun.

```py
!pip install -U accelerate
```

`device_map="auto"` ayarı, modelin önce en hızlı cihazlara (GPU'lar) otomatik olarak dağıtılmasını, ardından (varsa) daha yavaş cihazlara (CPU, sabit disk) gönderilmesini sağlar.

[`Pipeline`] yarı duyarlıklı ağırlıkları (torch.float16) destekler; bu önemli ölçüde hız kazandırır ve bellekten tasarruf sağlar. Çoğu modelde, özellikle büyük modellerde performans kaybı ihmal edilebilir düzeydedir. Donanımınız destekliyorsa, daha geniş bir aralık için torch.bfloat16'yı etkinleştirebilirsiniz.

> [!İPUCU]
> Girdiler dahili olarak torch.float16'ya dönüştürülür ve bu yalnızca PyTorch altyapısına sahip modellerde çalışır.

Son olarak, [`Pipeline`] bellek kullanımını daha da azaltmak için nicemlenmiş (quantized) modelleri de kabul eder. Öncelikle [bitsandbytes](https://hf.co/docs/bitsandbytes/installation) kütüphanesinin kurulu olduğundan emin olun ve ardından pipeline'daki `model_kwargs`'a `load_in_8bit=True` ekleyin.

```py
import torch
from transformers import pipeline, BitsAndBytesConfig

pipeline = pipeline(model="google/gemma-7b", torch_dtype=torch.bfloat16, device_map="auto", model_kwargs={"quantization_config": BitsAndBytesConfig(load_in_8bit=True)})
pipeline("iyi bir kek pişirmenin sırrı ")
[{'generated_text': 'iyi bir kek pişirmenin sırrı 1. doğru malzemeler 2. doğru'}]
```
