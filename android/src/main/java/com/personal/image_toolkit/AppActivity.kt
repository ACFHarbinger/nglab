package com.personal.nglab

import android.os.Bundle
import android.view.Menu
import android.view.MenuItem
import androidx.appcompat.app.AppCompatActivity
import androidx.navigation.findNavController
import androidx.navigation.fragment.NavHostFragment
import androidx.navigation.ui.AppBarConfiguration
import androidx.navigation.ui.navigateUp
import androidx.navigation.ui.setupActionBarWithNavController
import androidx.navigation.ui.setupWithNavController
import com.google.android.material.snackbar.Snackbar
import com.personal.nglab.databinding.ActivityAppBinding

class AppActivity : AppCompatActivity() {
    private lateinit var appBarConfiguration: AppBarConfiguration
    private lateinit var binding: ActivityAppBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityAppBinding.inflate(layoutInflater)
        setContentView(binding.root)
        setSupportActionBar(binding.appBarApp.toolbar)

        // FAB Action (Example: Quick Open Log)
        binding.appBarApp.fab?.setOnClickListener { view ->
            // Logic to show LogFragment could go here
            Snackbar.make(view, "Logs feature coming soon", Snackbar.LENGTH_LONG)
                .setAction("Action", null)
                .setAnchorView(R.id.fab).show()
        }

        // --- AUTHENTICATION CHECK ---
        // In a real app, check VaultManager state here.
        // If not logged in, show LoginFragment instead of NavHost.
        // For this example, we assume we proceed to the main UI.

        val navHostFragment =
            (supportFragmentManager.findFragmentById(R.id.nav_host_fragment_content_app) as NavHostFragment?)!!
        val navController = navHostFragment.navController

        // Update AppBarConfig with the new IDs defined in mobile_navigation.xml
        // Including all categories from PySide6: System Tools, Database, Web, Deep Learning
        val topLevelDestinations =
            setOf(
                R.id.nav_system_tools,
                R.id.nav_web_integration,
                R.id.nav_deep_learning,
            )

        binding.navView?.let {
            appBarConfiguration =
                AppBarConfiguration(
                    topLevelDestinations,
                    binding.drawerLayout,
                )
            setupActionBarWithNavController(navController, appBarConfiguration)
            it.setupWithNavController(navController)
        }

        // If you still have a BottomNav, link it here using the same IDs
        binding.appBarApp.contentApp.bottomNavView?.let {
            appBarConfiguration = AppBarConfiguration(topLevelDestinations)
            setupActionBarWithNavController(navController, appBarConfiguration)
            it.setupWithNavController(navController)
        }
    }

    override fun onCreateOptionsMenu(menu: Menu): Boolean {
        // Assume R.menu.overflow contains the items or is the intended overflow menu.
        menuInflater.inflate(R.menu.overflow, menu)
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        // The menu item ID for navigation must match the destination ID.
        return when (item.itemId) {
            R.id.nav_settings -> {
                val navController = findNavController(R.id.nav_host_fragment_content_app)
                navController.navigate(R.id.nav_settings)
                true
            }
            else -> super.onOptionsItemSelected(item)
        }
    }

    override fun onSupportNavigateUp(): Boolean {
        val navController = findNavController(R.id.nav_host_fragment_content_app)
        return navController.navigateUp(appBarConfiguration) || super.onSupportNavigateUp()
    }
}
