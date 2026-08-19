import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import Un0DemoPage from "./Un0DemoPage.tsx";

createRoot(document.getElementById("root")!).render(
	<StrictMode>
		<Un0DemoPage />
	</StrictMode>,
);
