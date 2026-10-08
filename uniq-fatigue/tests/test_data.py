import json
import tempfile
import shutil
import uuid
import unittest
from pathlib import Path
import numpy as np
from uniq_fatigue.prepare import synthetic,moving_average,resample,pmdata,wesad
from uniq_fatigue.data import load_dataset,context_text,pm_label,save_dataset
from uniq_fatigue.splits import make_split,audit_split
from uniq_fatigue.metrics import classification,bootstrap_subjects


class DataTests(unittest.TestCase):
    def setUp(self):
        self.root=Path.cwd()/"work"/("test_data_"+uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.path=self.root/"demo.npz"
        synthetic(self.path)
        self.x,self.m,self.records,self.channels=load_dataset(self.path)

    def tearDown(self):shutil.rmtree(self.root)

    def test_subject_split_and_seed(self):
        a=make_split(self.records,"subject",42)
        b=make_split(self.records,"subject",42)
        self.assertEqual(a,b)
        self.assertEqual(a["audit"]["cross_split_overlaps"],0)
        self.assertEqual(sum(len(a[k]) for k in ("train","val","test")),72)

    def test_overlap_rejected_and_chronological_purged(self):
        records=[]
        for i in range(30):
            r=dict(self.records[i]);r.update(id=str(i),subject="one",session="continuous",start=i*50,end=i*50+100)
            records.append(r)
        randomsplit=make_split(records,"random",42)
        self.assertGreater(randomsplit["audit"]["cross_split_overlaps"],0)
        with self.assertRaises(ValueError):audit_split(records,randomsplit)
        chronological=make_split(records,"chronological",42)
        self.assertEqual(chronological["audit"]["cross_split_overlaps"],0)
        self.assertGreater(chronological["audit"]["excluded"],0)

    def test_loso(self):
        for subject in sorted({r["subject"] for r in self.records}):
            split=make_split(self.records,"loso",42,subject)
            test={r["subject"] for r in self.records if r["id"] in split["test"]}
            self.assertEqual(test,{subject})

    def test_modality_dropout_removes_sensor_summary(self):
        t=context_text(self.x[0],self.m[0],self.records[0],self.channels,"sensor")
        self.assertNotIn("variance=",t)
        t=context_text(self.x[0],self.m[0],self.records[0],self.channels,"text")
        self.assertIn("Subjective log unavailable",t)
        self.assertNotIn(self.records[0]["text"],t)

    def test_label_map_is_not_silently_invented(self):
        with self.assertRaises(ValueError):pm_label(3,"native5")
        self.assertEqual([pm_label(i,"paper10") for i in [1,3,4,7,8,10]],[0,0,1,1,2,2])

    def test_metric_math(self):
        m=classification([0,0,1,1,2,2],[0,1,1,1,2,0])
        self.assertAlmostEqual(m["accuracy"],2/3)
        self.assertAlmostEqual(m["macro_f1"],(0.5+0.8+2/3)/3)
        self.assertAlmostEqual(m["kappa"],0.5)
        self.assertEqual(classification([0,1,2],[0,1,2])["kappa"],1.0)
        self.assertIsNone(classification([0],[0])["kappa"])

    def test_filter_and_resample(self):
        np.testing.assert_allclose(moving_average(np.ones(20)),np.ones(20))
        r=resample(np.arange(4),4,8,7)
        np.testing.assert_allclose(r,np.arange(7)*0.5)

    def test_native_pm_csv_import(self):
        raw=self.root/"raw"; (raw/"p01").mkdir(parents=True)
        (raw/"p01"/"sensors.csv").write_text("timestamp,heart_rate\n2020-01-01T00:00:00,60\n2020-01-01T00:00:10,70\n2020-01-01T00:00:20,80\n",encoding="utf-8")
        metadata=self.root/"intervals.csv"
        metadata.write_text("subject,session,start_iso,end_iso,fatigue,sleep_quality\np01,day1,2020-01-01T00:00:00,2020-01-01T00:00:21,4,poor\n",encoding="utf-8")
        with self.assertRaises(ValueError):pmdata(raw,metadata,self.root/"pm.npz")
        mapping=self.root/"map.json";mapping.write_text(json.dumps({"1":2,"2":2,"3":1,"4":0,"5":0}))
        pmdata(raw,metadata,self.root/"pm.npz",label_map=mapping,length=10)
        x,m,r,c=load_dataset(self.root/"pm.npz")
        self.assertEqual(r[0]["label"],0)
        self.assertNotIn("fatigue",r[0]["text"])
        self.assertEqual(c,["heart_rate"])

    def test_wesad_rejects_transition_windows(self):
        import pickle
        raw=self.root/"wesad";raw.mkdir()
        signal={"chest":{"ECG":np.ones((2100,1)),"Resp":np.ones((2100,1))},
                "wrist":{"EDA":np.ones((12,1)),"BVP":np.ones((192,1)),"TEMP":np.ones((12,1))}}
        labels=np.repeat([1,2,3],700)
        with (raw/"S2.pkl").open("wb") as f:pickle.dump({"signal":signal,"label":labels},f)
        out=self.root/"wesad.npz"
        with self.assertRaises(ValueError):wesad(raw,out)
        wesad(raw,out,trusted_pickle=True)
        x,m,r,c=load_dataset(out)
        self.assertEqual(len(x),3)
        self.assertEqual([s["label"] for s in r],[0,2,1])
        self.assertTrue(all("stress" not in s["text"].lower() for s in r))


if __name__=="__main__":unittest.main()
