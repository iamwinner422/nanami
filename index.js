const tf = require('@tensorflow/tfjs');
const use = require('@tensorflow-models/universal-sentence-encoder');

const trainData = require('./data.json');
const testData = require('./test.json');

const encodeData = async (data) => {
    const sentences = data.map(item => item.input.toLowerCase());
    const model = await use.load();
    return await model.embed(sentences);
};

const createOutputData = (data) => {
    return tf.tensor2d(data.map(item => [
        item.label === 'inspiring' ? 1 : 0,
        item.label === 'not-inspiring' ? 1 : 0,
    ]));
};

const model = tf.sequential();

model.add(tf.layers.dense({
    inputShape: [512],
    activation: 'sigmoid',
    units: 2,
}));

model.add(tf.layers.dense({
    activation: 'sigmoid',
    units: 2,
}));

model.add(tf.layers.dense({
    activation: 'sigmoid',
    units: 2,
}));

model.compile({
    loss: 'meanSquaredError',
    optimizer: tf.train.adam(0.06),
});

async function run() {
    try {
        const [trainingEmbeddings, testingEmbeddings] = await Promise.all([
            encodeData(trainData),
            encodeData(testData)
        ]);

        const outputData = createOutputData(trainData);

        await model.fit(trainingEmbeddings, outputData, { epochs: 200 });

        model.predict(testingEmbeddings).print();
    } catch (err) {
        console.error('Run Error:', err);
    }
}

run();
