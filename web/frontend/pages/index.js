import Head from 'next/head';
import {useEffect, useState} from 'react';
import styles from '../styles/Home.module.css';

const API_URL = 'http://repronets.boxkey.me/api/predict';

export default function Home() {
  const [wordToTranslate, setWordToTranslate] = useState('test');
  const [targetLanguage, setTargetLanguage] = useState('ara');
  const [numPredictions, setNumPredictions] = useState('2');
  const [translationData, setTranslationData] = useState({});

  useEffect(() => {
    const getData = async () => {
      const resp = await fetch(
        `${API_URL}?input=${wordToTranslate}&language=${targetLanguage}&beam=${numPredictions}&model=all`,
      );
      // const data = JSON.parse(
      //   '{"data":{"phonetisaurus":{"data":{"No.1":{"prob":1.313e-7,"tokens":"羅基"},"No.2":{"prob":6.24e-8,"tokens":"洛基"}},"status":200,"message":"Successfully made predictions"},"transformer":{"data":{"No.1":{"prob":0.207763698,"tokens":"罗基"},"No.2":{"prob":0.1878318166,"tokens":"羅基"}},"status":200,"message":"Successfully made predictions"}},"status":200}',
      // ).data;
      const json = await resp.json();
      setTranslationData(json.data);
    };
    getData();
  }, [wordToTranslate, targetLanguage, numPredictions]);

  return (
    <div className={styles.container}>
      <Head>
        <title>ReproNETs</title>
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <main className={styles.main}>
        <p className={styles.prompt}>
          Hey ReproNETs, how would you say{' '}
          <input
            className={styles.textInput}
            type="text"
            onChange={(e) => setWordToTranslate(e.target.value)}
            value={wordToTranslate}
          />{' '}
          in{' '}
          <select
            className={styles.select}
            value={targetLanguage}
            onChange={(e) => setTargetLanguage(e.target.value)}
          >
            <option value="ara">Arabic</option>
            <option value="chi">Chinese</option>
            <option value="heb">Hebrew</option>
            <option value="jpn">Japanese</option>
            <option value="kor">Korean</option>
            <option value="rus">Russian</option>
          </select>
          ?
          <br />
          Make{' '}
          <input
            className={styles.textInput}
            type="text"
            onChange={(e) => setNumPredictions(e.target.value)}
            value={numPredictions}
          />{' '}
          predictions.
        </p>

        <p className={styles.response}>
          {translationData &&
          translationData.transformer &&
          translationData.phonetisaurus ? (
            <>
              Like so:
              <table>
                <thead className={styles.thead}>
                  <td>Rank</td>
                  <td>Phonetisaurus (Probability)</td>
                  <td>Transformer (Probability)</td>
                </thead>
                {Object.entries(translationData.transformer?.data).map(
                  (_, i) => {
                    const curTransfomerObj =
                      translationData.transformer?.data[`No.${i + 1}`];
                    const curPhonetisaurusObj =
                      translationData.phonetisaurus?.data[`No.${i + 1}`];
                    return (
                      <tr>
                        <td>{i + 1}</td>
                        <td>{`${curPhonetisaurusObj?.tokens} (${
                          curPhonetisaurusObj?.prob || 0
                        })`}</td>
                        <td>{`${curTransfomerObj?.tokens} (${
                          curTransfomerObj?.prob || 0
                        })`}</td>
                      </tr>
                    );
                  },
                )}
              </table>
            </>
          ) : null}
        </p>
      </main>

      <footer className={styles.footer}>
        <a
          href="https://vercel.com?utm_source=create-next-app&utm_medium=default-template&utm_campaign=create-next-app"
          target="_blank"
          rel="noopener noreferrer"
        >
          Powered by{' '}
        </a>
      </footer>
    </div>
  );
}
