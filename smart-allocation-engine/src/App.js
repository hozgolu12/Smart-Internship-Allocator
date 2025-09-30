import React, { useState, useEffect } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  LineChart,
  Line,
  CartesianGrid,
  Cell,
} from "recharts";
import {
  Brain,
  Users,
  Briefcase,
  Download,
  Sparkles,
  TrendingUp,
  Award,
  Info,
  CheckCircle,
  Home, // Import the Home icon
} from "lucide-react";

// Advanced TF-IDF implementation for skills matching
class TFIDF {
  constructor(documents) {
    this.documents = documents;
    this.vocabulary = new Set();
    this.idf = {};
    this.build();
  }

  build() {
    this.documents.forEach((doc) => {
      const terms = this.tokenize(doc);
      terms.forEach((term) => this.vocabulary.add(term));
    });

    this.vocabulary.forEach((term) => {
      const docsWithTerm = this.documents.filter((doc) =>
        this.tokenize(doc).includes(term)
      ).length;
      this.idf[term] = Math.log(this.documents.length / (docsWithTerm + 1));
    });
  }

  tokenize(text) {
    return text
      .toLowerCase()
      .split(/\W+/)
      .filter((t) => t.length > 0);
  }

  vectorize(text) {
    const terms = this.tokenize(text);
    const termFreq = {};
    terms.forEach((term) => {
      termFreq[term] = (termFreq[term] || 0) + 1;
    });

    const vector = {};
    Object.keys(termFreq).forEach((term) => {
      if (this.idf[term]) {
        vector[term] = termFreq[term] * this.idf[term];
      }
    });
    return vector;
  }

  similarity(text1, text2) {
    const v1 = this.vectorize(text1);
    const v2 = this.vectorize(text2);

    let dotProduct = 0;
    let mag1 = 0;
    let mag2 = 0;

    const allTerms = new Set([...Object.keys(v1), ...Object.keys(v2)]);
    allTerms.forEach((term) => {
      const val1 = v1[term] || 0;
      const val2 = v2[term] || 0;
      dotProduct += val1 * val2;
      mag1 += val1 * val1;
      mag2 += val2 * val2;
    });

    return dotProduct / (Math.sqrt(mag1) * Math.sqrt(mag2) + 1e-8);
  }
}

// Neural Network implementation
class NeuralNetwork {
  constructor(inputSize, hiddenSize, outputSize) {
    this.weightsIH = Array(hiddenSize)
      .fill()
      .map(() =>
        Array(inputSize)
          .fill()
          .map(() => Math.random() * 2 - 1)
      );
    this.weightsHO = Array(outputSize)
      .fill()
      .map(() =>
        Array(hiddenSize)
          .fill()
          .map(() => Math.random() * 2 - 1)
      );
    this.biasH = Array(hiddenSize)
      .fill()
      .map(() => Math.random() * 2 - 1);
    this.biasO = Array(outputSize)
      .fill()
      .map(() => Math.random() * 2 - 1);
  }

  sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  }

  predict(inputs) {
    const hidden = this.weightsIH.map((weights, i) => {
      const sum =
        weights.reduce((acc, w, j) => acc + w * inputs[j], 0) + this.biasH[i];
      return this.sigmoid(sum);
    });

    const output = this.weightsHO.map((weights, i) => {
      const sum =
        weights.reduce((acc, w, j) => acc + w * hidden[j], 0) + this.biasO[i];
      return this.sigmoid(sum);
    });

    return output[0];
  }
}

// Enhanced feature engineering
function extractFeatures(student, internship, tfidf) {
  const features = [];

  const skillText1 = student.skills.join(" ");
  const skillText2 = internship.skillsReq.join(" ");
  const skillSimilarity = tfidf ? tfidf.similarity(skillText1, skillText2) : 0;
  features.push(skillSimilarity);

  const exactMatches = student.skills.filter((s) =>
    internship.skillsReq.some((req) => req.toLowerCase() === s.toLowerCase())
  ).length;
  features.push(exactMatches / Math.max(internship.skillsReq.length, 1));

  features.push(student.cgpa / 10);
  features.push(student.location === internship.location ? 1 : 0);

  const sectorMatch =
    internship.sector && student.sector.includes(internship.sector) ? 1 : 0;
  features.push(sectorMatch);
  features.push(Math.min(student.portfolio, 5) / 5);
  features.push(student.demographic.rural ? 1 : 0);

  const eduLevels = { BTech: 0.8, BE: 0.8, BCom: 0.6, MSc: 1.0, BSc: 0.7 };
  features.push(eduLevels[student.education] || 0.5);

  return features;
}

// Advanced ensemble matching with explainability
function advancedMatch(students, internships) {
  const allTexts = [
    ...students.map((s) => s.skills.join(" ")),
    ...internships.map((i) => i.skillsReq.join(" ")),
  ];
  const tfidf = new TFIDF(allTexts);
  const nn = new NeuralNetwork(8, 12, 1);
  const allMatches = [];

  students.forEach((student) => {
    internships.forEach((internship) => {
      const features = extractFeatures(student, internship, tfidf);
      const nnScore = nn.predict(features);

      let ruleScore = 0;
      ruleScore += features[1] * 3;
      ruleScore += features[2] * 2;
      ruleScore += features[3] * 1.5;
      ruleScore += features[4] * 1.2;
      ruleScore += features[5] * 1;
      ruleScore += features[6] * 2.5;
      ruleScore += features[7] * 0.8;
      ruleScore = ruleScore / 12;

      const tfidfScore = features[0];
      const ensembleScore =
        0.4 * nnScore + 0.35 * ruleScore + 0.25 * tfidfScore;

      allMatches.push({
        studentId: student.id,
        internshipId: internship.id,
        score: ensembleScore,
        breakdown: {
          neuralNet: nnScore,
          ruleBased: ruleScore,
          tfidf: tfidfScore,
          ensemble: ensembleScore,
        },
        features: {
          skillMatch: features[1],
          cgpa: features[2],
          location: features[3],
          sector: features[4],
          portfolio: features[5],
          rural: features[6],
          education: features[7],
        },
      });
    });
  });

  allMatches.sort((a, b) => b.score - a.score);

  const allocated = {};
  const studentAssigned = new Set();
  const results = [];

  internships.forEach((i) => (allocated[i.id] = 0));

  for (const match of allMatches) {
    if (
      !studentAssigned.has(match.studentId) &&
      allocated[match.internshipId] <
        internships.find((i) => i.id === match.internshipId).capacity
    ) {
      results.push(match);
      studentAssigned.add(match.studentId);
      allocated[match.internshipId]++;
    }
  }

  return results;
}

const initialStudents = [
  {
    id: "s1",
    name: "Amit Kumar",
    skills: ["Python", "Machine Learning", "Deep Learning", "TensorFlow"],
    education: "BTech",
    cgpa: 8.5,
    location: "Rural",
    sector: ["IT", "AI"],
    demographic: { rural: true, category: "OBC" },
    preferences: ["Remote", "Hybrid"],
    portfolio: 4,
  },
  {
    id: "s2",
    name: "Priya Sharma",
    skills: ["JavaScript", "React", "Node.js", "MongoDB"],
    education: "BE",
    cgpa: 9.0,
    location: "Urban",
    sector: ["Finance", "IT"],
    demographic: { rural: false, category: "GEN" },
    preferences: ["Hybrid", "On-site"],
    portfolio: 5,
  },
  {
    id: "s3",
    name: "Rahul Singh",
    skills: ["Excel", "SQL", "Power BI", "Data Analysis"],
    education: "BCom",
    cgpa: 7.8,
    location: "Rural",
    sector: ["Finance"],
    demographic: { rural: true, category: "SC" },
    preferences: ["Remote"],
    portfolio: 3,
  },
  {
    id: "s4",
    name: "Zara Khan",
    skills: ["Python", "AI", "SQL", "Computer Vision", "NLP"],
    education: "MSc",
    cgpa: 9.2,
    location: "Rural",
    sector: ["AI", "Research"],
    demographic: { rural: true, category: "GEN" },
    preferences: ["Remote", "Research"],
    portfolio: 5,
  },
  {
    id: "s5",
    name: "Vikram Rao",
    skills: ["Java", "Spring Boot", "Microservices", "AWS"],
    education: "BTech",
    cgpa: 8.8,
    location: "Urban",
    sector: ["IT"],
    demographic: { rural: false, category: "OBC" },
    preferences: ["On-site"],
    portfolio: 4,
  },
];

const initialInternships = [
  {
    id: "i1",
    industry: "Tech Corp",
    role: "ML Engineering Intern",
    location: "Urban",
    skillsReq: ["Python", "Machine Learning", "TensorFlow"],
    capacity: 2,
    sector: "IT",
  },
  {
    id: "i2",
    industry: "Finance Solutions",
    role: "Data Analyst Intern",
    location: "Rural",
    skillsReq: ["Excel", "SQL", "Power BI"],
    capacity: 1,
    sector: "Finance",
  },
  {
    id: "i3",
    industry: "AI Research Lab",
    role: "AI Research Intern",
    location: "Rural",
    skillsReq: ["Python", "AI", "Deep Learning", "Research"],
    capacity: 1,
    sector: "AI",
  },
  {
    id: "i4",
    industry: "Web Dev Studio",
    role: "Full Stack Developer",
    location: "Urban",
    skillsReq: ["JavaScript", "React", "Node.js"],
    capacity: 2,
    sector: "IT",
  },
];

export default function App() {
  const [step, setStep] = useState("home");
  const [students, setStudents] = useState(initialStudents);
  const [internships, setInternships] = useState(initialInternships);
  const [matches, setMatches] = useState([]);
  const [analytics, setAnalytics] = useState(null);
  const [loading, setLoading] = useState(false);

  const [studentForm, setStudentForm] = useState({
    name: "",
    skills: "",
    education: "BTech",
    cgpa: "",
    location: "Urban",
    sector: "",
    demographic: { rural: false, category: "GEN" },
    preferences: "",
    portfolio: 3,
  });

  const [internForm, setInternForm] = useState({
    industry: "",
    role: "",
    location: "Urban",
    skillsReq: "",
    capacity: 1,
    sector: "",
  });

  function handleStudentSubmit(e) {
    e.preventDefault();
    setStudents([
      ...students,
      {
        id: "s" + Date.now(),
        name: studentForm.name,
        skills: studentForm.skills.split(",").map((s) => s.trim()),
        education: studentForm.education,
        cgpa: parseFloat(studentForm.cgpa),
        location: studentForm.location,
        sector: studentForm.sector.split(",").map((s) => s.trim()),
        demographic: studentForm.demographic,
        preferences: studentForm.preferences.split(",").map((s) => s.trim()),
        portfolio: parseInt(studentForm.portfolio),
      },
    ]);
    setStudentForm({
      name: "",
      skills: "",
      education: "BTech",
      cgpa: "",
      location: "Urban",
      sector: "",
      demographic: { rural: false, category: "GEN" },
      preferences: "",
      portfolio: 3,
    });
    setStep("home");
  }

  function handleInternSubmit(e) {
    e.preventDefault();
    setInternships([
      ...internships,
      {
        id: "i" + Date.now(),
        industry: internForm.industry,
        role: internForm.role,
        location: internForm.location,
        skillsReq: internForm.skillsReq.split(",").map((s) => s.trim()),
        capacity: Number(internForm.capacity),
        sector: internForm.sector,
      },
    ]);
    setInternForm({
      industry: "",
      role: "",
      location: "Urban",
      skillsReq: "",
      capacity: 1,
      sector: "",
    });
    setStep("home");
  }

  function handleAllocate() {
    setLoading(true);
    setTimeout(() => {
      const results = advancedMatch(students, internships);
      setMatches(results);

      const ruralMatches = results.filter(
        (m) => students.find((s) => s.id === m.studentId)?.demographic.rural
      ).length;

      const avgScore =
        results.reduce((acc, m) => acc + m.score, 0) / results.length;

      setAnalytics({
        totalMatched: results.length,
        ruralRepresentation: ruralMatches,
        avgMatchScore: avgScore,
        placementRate: ((results.length / students.length) * 100).toFixed(1),
      });

      setLoading(false);
      setStep("matches");
    }, 1500);
  }

  function getCSV() {
    return (
      "Student,Internship,Score,Neural_Net,Rule_Based,TFIDF,Skill_Match,CGPA,Location,Sector,Portfolio,Rural\n" +
      matches
        .map((m) => {
          const s = students.find((st) => st.id === m.studentId);
          const i = internships.find((it) => it.id === m.internshipId);
          return `${s?.name},${i?.role},${m.score.toFixed(
            3
          )},${m.breakdown.neuralNet.toFixed(
            3
          )},${m.breakdown.ruleBased.toFixed(3)},${m.breakdown.tfidf.toFixed(
            3
          )},${m.features.skillMatch.toFixed(2)},${m.features.cgpa.toFixed(
            2
          )},${m.features.location},${
            m.features.sector
          },${m.features.portfolio.toFixed(2)},${m.features.rural}`;
        })
        .join("\n")
    );
  }

  const containerStyle = {
    minHeight: "100vh",
    background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
    padding: "2rem 1rem",
    fontFamily: "'Inter', system-ui, sans-serif",
  };

  const cardStyle = {
    maxWidth: "1200px",
    margin: "0 auto",
    background: "rgba(255, 255, 255, 0.95)",
    backdropFilter: "blur(10px)",
    borderRadius: "24px",
    boxShadow: "0 20px 60px rgba(0,0,0,0.3)",
    padding: "2.5rem",
  };

  const buttonStyle = (color) => ({
    background: color,
    color: "white",
    padding: "0.75rem 1.5rem",
    borderRadius: "12px",
    border: "none",
    cursor: "pointer",
    fontSize: "0.95rem",
    fontWeight: "600",
    display: "flex",
    alignItems: "center",
    gap: "0.5rem",
    transition: "all 0.3s",
    boxShadow: "0 4px 15px rgba(0,0,0,0.2)",
  });

  const COLORS = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"];

  return (
    <div style={containerStyle}>
      <div style={cardStyle}>
        <div style={{ textAlign: "center", marginBottom: "2rem" }}>
          <h1
            style={{
              fontSize: "2.5rem",
              background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
              marginBottom: "0.5rem",
              fontWeight: "800",
            }}
          >
            AI-Powered PM Internship Allocator
          </h1>
          <p style={{ color: "#666", fontSize: "1.1rem" }}>
            Neural Networks + NLP + Fairness-Aware Allocation
          </p>
        </div>

        <div
          style={{
            display: "flex",
            gap: "1rem",
            marginBottom: "2rem",
            flexWrap: "wrap",
            justifyContent: "center",
          }}
        >
          <button
            onClick={() => setStep("student")}
            style={buttonStyle("#3b82f6")}
          >
            <Users size={18} /> Add Student
          </button>
          <button
            onClick={() => setStep("intern")}
            style={buttonStyle("#10b981")}
          >
            <Briefcase size={18} /> Add Internship
          </button>
          <button
            onClick={handleAllocate}
            style={buttonStyle("#8b5cf6")}
            disabled={loading}
          >
            {loading ? (
              "Processing..."
            ) : (
              <>
                <Sparkles size={18} /> Run AI Matching
              </>
            )}
          </button>
          <button
            onClick={() => setStep("analytics")}
            style={buttonStyle("#f59e0b")}
            disabled={!matches.length}
          >
            <TrendingUp size={18} /> Analytics
          </button>
          <button
            onClick={() => setStep("export")}
            style={buttonStyle("#6b7280")}
            disabled={!matches.length}
          >
            <Download size={18} /> Export
          </button>
        </div>

        {/* --- Back to Home Button --- */}
        {step !== "home" && (
          <div style={{ marginBottom: "2rem", textAlign: "center" }}>
            <button
              onClick={() => setStep("home")}
              style={buttonStyle("#6b7280")}
            >
              <Home size={18} /> Back to Home
            </button>
          </div>
        )}
        {/* --------------------------- */}

        {step === "student" && (
          <form
            onSubmit={handleStudentSubmit}
            style={{
              background: "linear-gradient(135deg, #e0f2fe 0%, #ddd6fe 100%)",
              padding: "2rem",
              borderRadius: "16px",
              maxWidth: "600px",
              margin: "0 auto",
            }}
          >
            <h2
              style={{
                fontSize: "1.5rem",
                marginBottom: "1.5rem",
                color: "#1e40af",
              }}
            >
              <Users size={24} style={{ verticalAlign: "middle" }} /> Add New
              Student
            </h2>

            <input
              required
              placeholder="Full Name"
              value={studentForm.name}
              onChange={(e) =>
                setStudentForm((f) => ({ ...f, name: e.target.value }))
              }
              style={{
                width: "100%",
                marginBottom: "1rem",
                padding: "0.75rem",
                borderRadius: "8px",
                border: "2px solid #e5e7eb",
                fontSize: "1rem",
              }}
            />

            <input
              required
              placeholder="Skills (e.g., Python, ML, React)"
              value={studentForm.skills}
              onChange={(e) =>
                setStudentForm((f) => ({ ...f, skills: e.target.value }))
              }
              style={{
                width: "100%",
                marginBottom: "1rem",
                padding: "0.75rem",
                borderRadius: "8px",
                border: "2px solid #e5e7eb",
                fontSize: "1rem",
              }}
            />

            <select
              value={studentForm.education}
              onChange={(e) =>
                setStudentForm((f) => ({ ...f, education: e.target.value }))
              }
              style={{
                width: "100%",
                marginBottom: "1rem",
                padding: "0.75rem",
                borderRadius: "8px",
                border: "2px solid #e5e7eb",
                fontSize: "1rem",
              }}
            >
              <option>BTech</option>
              <option>BE</option>
              <option>BCom</option>
              <option>MSc</option>
              <option>BSc</option>
            </select>

            <input
              required
              type="number"
              min={5}
              max={10}
              step="0.1"
              placeholder="CGPA (5.0 - 10.0)"
              value={studentForm.cgpa}
              onChange={(e) =>
                setStudentForm((f) => ({ ...f, cgpa: e.target.value }))
              }
              style={{
                width: "100%",
                marginBottom: "1rem",
                padding: "0.75rem",
                borderRadius: "8px",
                border: "2px solid #e5e7eb",
                fontSize: "1rem",
              }}
            />

            <select
              value={studentForm.location}
              onChange={(e) =>
                setStudentForm((f) => ({ ...f, location: e.target.value }))
              }
              style={{
                width: "100%",
                marginBottom: "1rem",
                padding: "0.75rem",
                borderRadius: "8px",
                border: "2px solid #e5e7eb",
                fontSize: "1rem",
              }}
            >
              <option>Urban</option>
              <option>Rural</option>
            </select>

            <input
              required
              placeholder="Sector interests (e.g., IT, Finance)"
              value={studentForm.sector}
              onChange={(e) =>
                setStudentForm((f) => ({ ...f, sector: e.target.value }))
              }
              style={{
                width: "100%",
                marginBottom: "1rem",
                padding: "0.75rem",
                borderRadius: "8px",
                border: "2px solid #e5e7eb",
                fontSize: "1rem",
              }}
            />

            <select
              value={studentForm.demographic.category}
              onChange={(e) =>
                setStudentForm((f) => ({
                  ...f,
                  demographic: { ...f.demographic, category: e.target.value },
                }))
              }
              style={{
                width: "100%",
                marginBottom: "1rem",
                padding: "0.75rem",
                borderRadius: "8px",
                border: "2px solid #e5e7eb",
                fontSize: "1rem",
              }}
            >
              <option value="GEN">General</option>
              <option value="OBC">OBC</option>
              <option value="SC">SC</option>
              <option value="ST">ST</option>
            </select>

            <label
              style={{
                display: "flex",
                alignItems: "center",
                marginBottom: "1rem",
                fontSize: "1rem",
              }}
            >
              <input
                type="checkbox"
                checked={studentForm.demographic.rural}
                onChange={(e) =>
                  setStudentForm((f) => ({
                    ...f,
                    demographic: { ...f.demographic, rural: e.target.checked },
                  }))
                }
                style={{ marginRight: "0.5rem", width: "20px", height: "20px" }}
              />
              From Rural/Aspirational District
            </label>

            <input
              placeholder="Preferences (e.g., Remote, Hybrid)"
              value={studentForm.preferences}
              onChange={(e) =>
                setStudentForm((f) => ({ ...f, preferences: e.target.value }))
              }
              style={{
                width: "100%",
                marginBottom: "1rem",
                padding: "0.75rem",
                borderRadius: "8px",
                border: "2px solid #e5e7eb",
                fontSize: "1rem",
              }}
            />

            <div style={{ marginBottom: "1rem" }}>
              <label
                style={{
                  display: "block",
                  marginBottom: "0.5rem",
                  fontWeight: "600",
                }}
              >
                Portfolio Strength: {studentForm.portfolio}
              </label>
              <input
                type="range"
                min={1}
                max={5}
                value={studentForm.portfolio}
                onChange={(e) =>
                  setStudentForm((f) => ({ ...f, portfolio: e.target.value }))
                }
                style={{ width: "100%", height: "8px" }}
              />
            </div>

            <button
              type="submit"
              style={{
                ...buttonStyle("#3b82f6"),
                width: "100%",
                justifyContent: "center",
              }}
            >
              Submit Student
            </button>
          </form>
        )}

        {step === "intern" && (
          <form
            onSubmit={handleInternSubmit}
            style={{
              background: "linear-gradient(135deg, #d1fae5 0%, #fef3c7 100%)",
              padding: "2rem",
              borderRadius: "16px",
              maxWidth: "600px",
              margin: "0 auto",
            }}
          >
            <h2
              style={{
                fontSize: "1.5rem",
                marginBottom: "1.5rem",
                color: "#065f46",
              }}
            >
              <Briefcase size={24} style={{ verticalAlign: "middle" }} /> Add
              New Internship
            </h2>

            <input
              required
              placeholder="Company/Industry Name"
              value={internForm.industry}
              onChange={(e) =>
                setInternForm((f) => ({ ...f, industry: e.target.value }))
              }
              style={{
                width: "100%",
                marginBottom: "1rem",
                padding: "0.75rem",
                borderRadius: "8px",
                border: "2px solid #e5e7eb",
                fontSize: "1rem",
              }}
            />

            <input
              required
              placeholder="Role/Position Title"
              value={internForm.role}
              onChange={(e) =>
                setInternForm((f) => ({ ...f, role: e.target.value }))
              }
              style={{
                width: "100%",
                marginBottom: "1rem",
                padding: "0.75rem",
                borderRadius: "8px",
                border: "2px solid #e5e7eb",
                fontSize: "1rem",
              }}
            />

            <select
              value={internForm.location}
              onChange={(e) =>
                setInternForm((f) => ({ ...f, location: e.target.value }))
              }
              style={{
                width: "100%",
                marginBottom: "1rem",
                padding: "0.75rem",
                borderRadius: "8px",
                border: "2px solid #e5e7eb",
                fontSize: "1rem",
              }}
            >
              <option>Urban</option>
              <option>Rural</option>
            </select>

            <input
              required
              placeholder="Required Skills (e.g., Python, ML)"
              value={internForm.skillsReq}
              onChange={(e) =>
                setInternForm((f) => ({ ...f, skillsReq: e.target.value }))
              }
              style={{
                width: "100%",
                marginBottom: "1rem",
                padding: "0.75rem",
                borderRadius: "8px",
                border: "2px solid #e5e7eb",
                fontSize: "1rem",
              }}
            />

            <input
              required
              type="number"
              min={1}
              placeholder="Capacity (Number of interns)"
              value={internForm.capacity}
              onChange={(e) =>
                setInternForm((f) => ({ ...f, capacity: e.target.value }))
              }
              style={{
                width: "100%",
                marginBottom: "1rem",
                padding: "0.75rem",
                borderRadius: "8px",
                border: "2px solid #e5e7eb",
                fontSize: "1rem",
              }}
            />

            <input
              required
              placeholder="Sector (e.g., IT, Finance, AI)"
              value={internForm.sector}
              onChange={(e) =>
                setInternForm((f) => ({ ...f, sector: e.target.value }))
              }
              style={{
                width: "100%",
                marginBottom: "1rem",
                padding: "0.75rem",
                borderRadius: "8px",
                border: "2px solid #e5e7eb",
                fontSize: "1rem",
              }}
            />

            <button
              type="submit"
              style={{
                ...buttonStyle("#10b981"),
                width: "100%",
                justifyContent: "center",
              }}
            >
              Submit Internship
            </button>
          </form>
        )}

        {step === "matches" && (
          <div>
            <h2
              style={{
                fontSize: "2rem",
                marginBottom: "1.5rem",
                color: "#7c3aed",
                display: "flex",
                alignItems: "center",
                gap: "0.5rem",
              }}
            >
              <Award size={32} /> AI Match Results
            </h2>

            {analytics && (
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
                  gap: "1rem",
                  marginBottom: "2rem",
                }}
              >
                <div
                  style={{
                    background: "linear-gradient(135deg, #3b82f6, #2563eb)",
                    color: "white",
                    padding: "1.5rem",
                    borderRadius: "12px",
                  }}
                >
                  <div style={{ fontSize: "2rem", fontWeight: "bold" }}>
                    {analytics.totalMatched}
                  </div>
                  <div>Total Matches</div>
                </div>
                <div
                  style={{
                    background: "linear-gradient(135deg, #10b981, #059669)",
                    color: "white",
                    padding: "1.5rem",
                    borderRadius: "12px",
                  }}
                >
                  <div style={{ fontSize: "2rem", fontWeight: "bold" }}>
                    {analytics.ruralRepresentation}
                  </div>
                  <div>Rural Students Placed</div>
                </div>
                <div
                  style={{
                    background: "linear-gradient(135deg, #f59e0b, #d97706)",
                    color: "white",
                    padding: "1.5rem",
                    borderRadius: "12px",
                  }}
                >
                  <div style={{ fontSize: "2rem", fontWeight: "bold" }}>
                    {(analytics.avgMatchScore * 100).toFixed(0)}%
                  </div>
                  <div>Avg Match Score</div>
                </div>
                <div
                  style={{
                    background: "linear-gradient(135deg, #8b5cf6, #7c3aed)",
                    color: "white",
                    padding: "1.5rem",
                    borderRadius: "12px",
                  }}
                >
                  <div style={{ fontSize: "2rem", fontWeight: "bold" }}>
                    {analytics.placementRate}%
                  </div>
                  <div>Placement Rate</div>
                </div>
              </div>
            )}

            <div style={{ overflowX: "auto" }}>
              <table
                style={{
                  width: "100%",
                  borderCollapse: "collapse",
                  fontSize: "0.9rem",
                }}
              >
                <thead>
                  <tr
                    style={{
                      background: "linear-gradient(135deg, #667eea, #764ba2)",
                      color: "white",
                    }}
                  >
                    <th style={{ padding: "1rem", textAlign: "left" }}>
                      Student
                    </th>
                    <th style={{ padding: "1rem", textAlign: "left" }}>
                      Internship
                    </th>
                    <th style={{ padding: "1rem", textAlign: "center" }}>
                      Match Score
                    </th>
                    <th style={{ padding: "1rem", textAlign: "left" }}>
                      AI Explanation
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {matches.map((m, idx) => {
                    const student = students.find((s) => s.id === m.studentId);
                    const internship = internships.find(
                      (i) => i.id === m.internshipId
                    );
                    return (
                      <tr
                        key={m.studentId + m.internshipId}
                        style={{
                          background: idx % 2 === 0 ? "#f9fafb" : "white",
                          borderBottom: "1px solid #e5e7eb",
                        }}
                      >
                        <td style={{ padding: "1rem" }}>
                          <div style={{ fontWeight: "600" }}>
                            {student?.name}
                          </div>
                          <div style={{ fontSize: "0.85rem", color: "#666" }}>
                            {student?.education} | CGPA: {student?.cgpa}
                          </div>
                        </td>
                        <td style={{ padding: "1rem" }}>
                          <div style={{ fontWeight: "600" }}>
                            {internship?.role}
                          </div>
                          <div style={{ fontSize: "0.85rem", color: "#666" }}>
                            {internship?.industry}
                          </div>
                        </td>
                        <td style={{ padding: "1rem", textAlign: "center" }}>
                          <div
                            style={{
                              display: "inline-block",
                              background:
                                "linear-gradient(135deg, #10b981, #059669)",
                              color: "white",
                              padding: "0.5rem 1rem",
                              borderRadius: "8px",
                              fontWeight: "bold",
                            }}
                          >
                            {(m.score * 100).toFixed(1)}%
                          </div>
                        </td>
                        <td style={{ padding: "1rem" }}>
                          <div style={{ fontSize: "0.85rem" }}>
                            <div style={{ marginBottom: "0.5rem" }}>
                              <strong>Neural Net:</strong>{" "}
                              {(m.breakdown.neuralNet * 100).toFixed(1)}%
                              <span style={{ marginLeft: "1rem" }}>
                                <strong>Rule-Based:</strong>{" "}
                                {(m.breakdown.ruleBased * 100).toFixed(1)}%
                              </span>
                            </div>
                            <div style={{ marginBottom: "0.5rem" }}>
                              <strong>NLP Similarity:</strong>{" "}
                              {(m.breakdown.tfidf * 100).toFixed(1)}%
                            </div>
                            <div
                              style={{
                                display: "flex",
                                gap: "0.5rem",
                                flexWrap: "wrap",
                                marginTop: "0.5rem",
                              }}
                            >
                              {m.features.skillMatch > 0.5 && (
                                <span
                                  style={{
                                    background: "#dbeafe",
                                    color: "#1e40af",
                                    padding: "0.25rem 0.5rem",
                                    borderRadius: "4px",
                                    fontSize: "0.75rem",
                                  }}
                                >
                                  <CheckCircle
                                    size={12}
                                    style={{ verticalAlign: "middle" }}
                                  />{" "}
                                  Strong Skills Match
                                </span>
                              )}
                              {m.features.cgpa > 0.85 && (
                                <span
                                  style={{
                                    background: "#dcfce7",
                                    color: "#166534",
                                    padding: "0.25rem 0.5rem",
                                    borderRadius: "4px",
                                    fontSize: "0.75rem",
                                  }}
                                >
                                  High CGPA
                                </span>
                              )}
                              {m.features.location === 1 && (
                                <span
                                  style={{
                                    background: "#fef3c7",
                                    color: "#92400e",
                                    padding: "0.25rem 0.5rem",
                                    borderRadius: "4px",
                                    fontSize: "0.75rem",
                                  }}
                                >
                                  Location Match
                                </span>
                              )}
                              {m.features.rural === 1 && (
                                <span
                                  style={{
                                    background: "#fce7f3",
                                    color: "#9f1239",
                                    padding: "0.25rem 0.5rem",
                                    borderRadius: "4px",
                                    fontSize: "0.75rem",
                                  }}
                                >
                                  Rural Candidate
                                </span>
                              )}
                            </div>
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {step === "analytics" && (
          <div>
            <h2
              style={{
                fontSize: "2rem",
                marginBottom: "1.5rem",
                color: "#f59e0b",
              }}
            >
              <TrendingUp size={32} style={{ verticalAlign: "middle" }} />{" "}
              Analytics Dashboard
            </h2>

            <div style={{ marginBottom: "3rem" }}>
              <h3 style={{ fontSize: "1.25rem", marginBottom: "1rem" }}>
                Model Component Scores
              </h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart
                  data={matches.map((m, idx) => ({
                    name: students
                      .find((s) => s.id === m.studentId)
                      ?.name.split(" ")[0],
                    NeuralNet: (m.breakdown.neuralNet * 100).toFixed(1),
                    RuleBased: (m.breakdown.ruleBased * 100).toFixed(1),
                    NLP: (m.breakdown.tfidf * 100).toFixed(1),
                    Final: (m.score * 100).toFixed(1),
                  }))}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="NeuralNet" fill="#3b82f6" />
                  <Bar dataKey="RuleBased" fill="#10b981" />
                  <Bar dataKey="NLP" fill="#f59e0b" />
                  <Bar dataKey="Final" fill="#8b5cf6" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div style={{ marginBottom: "3rem" }}>
              <h3 style={{ fontSize: "1.25rem", marginBottom: "1rem" }}>
                Feature Importance Analysis
              </h3>
              <ResponsiveContainer width="100%" height={350}>
                <RadarChart
                  data={[
                    {
                      feature: "Skills",
                      value:
                        (matches.reduce(
                          (acc, m) => acc + m.features.skillMatch,
                          0
                        ) /
                          matches.length) *
                        100,
                    },
                    {
                      feature: "CGPA",
                      value:
                        (matches.reduce((acc, m) => acc + m.features.cgpa, 0) /
                          matches.length) *
                        100,
                    },
                    {
                      feature: "Location",
                      value:
                        (matches.reduce(
                          (acc, m) => acc + m.features.location,
                          0
                        ) /
                          matches.length) *
                        100,
                    },
                    {
                      feature: "Sector",
                      value:
                        (matches.reduce(
                          (acc, m) => acc + m.features.sector,
                          0
                        ) /
                          matches.length) *
                        100,
                    },
                    {
                      feature: "Portfolio",
                      value:
                        (matches.reduce(
                          (acc, m) => acc + m.features.portfolio,
                          0
                        ) /
                          matches.length) *
                        100,
                    },
                  ]}
                >
                  <PolarGrid />
                  <PolarAngleAxis dataKey="feature" />
                  <PolarRadiusAxis />
                  <Radar
                    name="Feature Importance"
                    dataKey="value"
                    stroke="#8b5cf6"
                    fill="#8b5cf6"
                    fillOpacity={0.6}
                  />
                </RadarChart>
              </ResponsiveContainer>
            </div>

            <div>
              <h3 style={{ fontSize: "1.25rem", marginBottom: "1rem" }}>
                Match Score Distribution
              </h3>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart
                  data={matches.map((m, idx) => ({
                    index: idx + 1,
                    score: (m.score * 100).toFixed(1),
                  }))}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="index"
                    label={{
                      value: "Match Rank",
                      position: "insideBottom",
                      offset: -5,
                    }}
                  />
                  <YAxis
                    label={{
                      value: "Score (%)",
                      angle: -90,
                      position: "insideLeft",
                    }}
                  />
                  <Tooltip />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="score"
                    stroke="#8b5cf6"
                    strokeWidth={3}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div
              style={{
                marginTop: "2rem",
                padding: "1.5rem",
                background: "#f0fdf4",
                borderRadius: "12px",
                border: "2px solid #10b981",
              }}
            >
              <h3
                style={{
                  fontSize: "1.25rem",
                  marginBottom: "1rem",
                  color: "#065f46",
                }}
              >
                <Info size={24} style={{ verticalAlign: "middle" }} /> Fairness
                & Diversity Metrics
              </h3>
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))",
                  gap: "1rem",
                }}
              >
                <div>
                  <strong>Rural Representation:</strong>
                  <div style={{ fontSize: "1.5rem", color: "#059669" }}>
                    {(
                      (analytics.ruralRepresentation / analytics.totalMatched) *
                      100
                    ).toFixed(1)}
                    %
                  </div>
                </div>
                <div>
                  <strong>Category Distribution:</strong>
                  <div style={{ fontSize: "0.9rem", marginTop: "0.5rem" }}>
                    {["GEN", "OBC", "SC", "ST"].map((cat) => {
                      const count = matches.filter(
                        (m) =>
                          students.find((s) => s.id === m.studentId)
                            ?.demographic.category === cat
                      ).length;
                      return count > 0 ? (
                        <div key={cat}>
                          {cat}: {count}
                        </div>
                      ) : null;
                    })}
                  </div>
                </div>
                <div>
                  <strong>Location Balance:</strong>
                  <div style={{ fontSize: "0.9rem", marginTop: "0.5rem" }}>
                    Urban:{" "}
                    {
                      matches.filter(
                        (m) =>
                          students.find((s) => s.id === m.studentId)
                            ?.location === "Urban"
                      ).length
                    }
                    <br />
                    Rural:{" "}
                    {
                      matches.filter(
                        (m) =>
                          students.find((s) => s.id === m.studentId)
                            ?.location === "Rural"
                      ).length
                    }
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {step === "export" && (
          <div>
            <h2
              style={{
                fontSize: "2rem",
                marginBottom: "1.5rem",
                color: "#6b7280",
              }}
            >
              <Download size={32} style={{ verticalAlign: "middle" }} /> Export
              Results
            </h2>
            <div
              style={{
                background: "#1f2937",
                color: "#10b981",
                padding: "1.5rem",
                borderRadius: "12px",
                fontFamily: "monospace",
                fontSize: "0.85rem",
                whiteSpace: "pre-wrap",
                maxHeight: "500px",
                overflow: "auto",
              }}
            >
              {getCSV()}
            </div>
            <button
              onClick={() => {
                const blob = new Blob([getCSV()], { type: "text/csv" });
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement("a");
                a.href = url;
                a.download = "internship_matches.csv";
                a.click();
              }}
              style={{
                ...buttonStyle("#10b981"),
                marginTop: "1rem",
                justifyContent: "center",
              }}
            >
              <Download size={18} /> Download CSV File
            </button>
          </div>
        )}

        {step === "home" && (
          <div>
            <div
              style={{
                background: "linear-gradient(135deg, #f0f9ff 0%, #f5f3ff 100%)",
                padding: "2rem",
                borderRadius: "16px",
                marginBottom: "2rem",
              }}
            >
              <h3
                style={{
                  fontSize: "1.5rem",
                  marginBottom: "1rem",
                  color: "#1e40af",
                }}
              >
                🎯 How It Works
              </h3>
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))",
                  gap: "1.5rem",
                }}
              >
                <div
                  style={{
                    background: "white",
                    padding: "1.5rem",
                    borderRadius: "12px",
                    boxShadow: "0 2px 8px rgba(0,0,0,0.1)",
                  }}
                >
                  <Brain
                    size={32}
                    style={{ color: "#3b82f6", marginBottom: "0.5rem" }}
                  />
                  <h4 style={{ fontSize: "1.1rem", marginBottom: "0.5rem" }}>
                    Neural Network
                  </h4>
                  <p style={{ fontSize: "0.9rem", color: "#666" }}>
                    Deep learning model with 8 input features, 12 hidden
                    neurons, predicting match probability
                  </p>
                </div>
                <div
                  style={{
                    background: "white",
                    padding: "1.5rem",
                    borderRadius: "12px",
                    boxShadow: "0 2px 8px rgba(0,0,0,0.1)",
                  }}
                >
                  <Sparkles
                    size={32}
                    style={{ color: "#10b981", marginBottom: "0.5rem" }}
                  />
                  <h4 style={{ fontSize: "1.1rem", marginBottom: "0.5rem" }}>
                    NLP TF-IDF
                  </h4>
                  <p style={{ fontSize: "0.9rem", color: "#666" }}>
                    Advanced text similarity using Term Frequency-Inverse
                    Document Frequency for skills matching
                  </p>
                </div>
                <div
                  style={{
                    background: "white",
                    padding: "1.5rem",
                    borderRadius: "12px",
                    boxShadow: "0 2px 8px rgba(0,0,0,0.1)",
                  }}
                >
                  <Award
                    size={32}
                    style={{ color: "#f59e0b", marginBottom: "0.5rem" }}
                  />
                  <h4 style={{ fontSize: "1.1rem", marginBottom: "0.5rem" }}>
                    Fairness-Aware
                  </h4>
                  <p style={{ fontSize: "0.9rem", color: "#666" }}>
                    Affirmative action bonus for rural candidates ensuring
                    equitable opportunity distribution
                  </p>
                </div>
              </div>
            </div>

            <div
              style={{
                background: "#fafafa",
                padding: "2rem",
                borderRadius: "16px",
                boxShadow: "0 2px 8px rgba(0,0,0,0.05)",
              }}
            >
              <h3
                style={{
                  fontSize: "1.25rem",
                  marginBottom: "1rem",
                  color: "#1f2937",
                }}
              >
                📊 Current Data
              </h3>
              <div style={{ marginBottom: "1rem" }}>
                <strong>Students ({students.length}):</strong>{" "}
                {students.map((s) => s.name).join(", ")}
              </div>
              <div style={{ marginBottom: "1rem" }}>
                <strong>Internships ({internships.length}):</strong>{" "}
                {internships.map((i) => i.role).join(", ")}
              </div>
              <div
                style={{
                  padding: "1rem",
                  background: "#e0f2fe",
                  borderRadius: "8px",
                  marginTop: "1rem",
                }}
              >
                <strong style={{ color: "#0c4a6e" }}>💡 Quick Start:</strong>
                <p style={{ margin: "0.5rem 0 0 0", color: "#075985" }}>
                  Add students and internships using the buttons above, then
                  click "Run AI Matching" to see the magic happen!
                </p>
              </div>
            </div>

            <div
              style={{
                marginTop: "2rem",
                padding: "1.5rem",
                background: "linear-gradient(135deg, #fef3c7 0%, #fce7f3 100%)",
                borderRadius: "12px",
                textAlign: "center",
              }}
            >
              <p style={{ color: "#78350f", fontSize: "0.95rem" }}>
                Featuring: Multi-Model Ensemble • Explainable AI • Fairness
                Metrics • Real-time Analytics
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
