/*
 * This file configures the root project and defines all sub-modules.
 */

pluginManagement {
    repositories {
        google {
            content {
                includeGroupByRegex("com\\.android.*")
                includeGroupByRegex("com\\.google.*")
                includeGroupByRegex("androidx.*")
            }
        }
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

plugins {
    // Apply the foojay-resolver plugin to allow automatic download of JDKs
    id("org.gradle.toolchains.foojay-resolver-convention") version "1.0.0"
    // Apply the Version Catalog plugin itself. Use its ID and version here.
    // Assuming you use the built-in catalog, this might not be needed, but if you define plugins here, you need this.
    // id("com.gradle.toolchains.foojay-resolver-convention") version "1.0.0" // Already there
}

rootProject.name = "nglab"

// Android Application Module
include("android")